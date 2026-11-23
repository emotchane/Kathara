import io
import json
import os
import sys
import tarfile
from unittest import mock
from unittest.mock import Mock

import pytest

sys.path.insert(0, './')

from src.Kathara.manager.docker.DockerImage import DockerImage, SAVE_BASE_LABEL, SAVE_DELETIONS_LABEL, \
    DELETIONS_OVERFLOW, DELETIONS_FILE, MAX_DELETIONS_LABEL, encode_deletions, decode_deletions
from src.Kathara.manager.docker.DockerMachine import RESTORE_DIFF_META
from src.Kathara.manager.docker.DockerManager import DockerManager
from src.Kathara.model.Lab import Lab
from src.Kathara.exceptions import InvocationError, MachineNotFoundError, SaveLabError


@pytest.fixture()
@mock.patch("src.Kathara.manager.docker.DockerPlugin.DockerPlugin.check_and_download_plugin")
@mock.patch("docker.client.DockerClient")
@mock.patch("docker.from_env")
def docker_manager(mock_from_env, client_mock, mock_check_and_download_plugin):
    mock_check_and_download_plugin.return_value = True
    mock_from_env.return_value = client_mock
    return DockerManager()


@pytest.fixture()
def running_lab():
    lab = Lab("Default scenario")
    pc1 = lab.get_or_new_machine("pc1", **{'image': 'kathara/test1'})
    pc2 = lab.get_or_new_machine("pc2", **{'image': 'kathara/test2'})
    lab.connect_machine_to_link(pc1.name, "A")
    lab.connect_machine_to_link(pc2.name, "A")
    for machine in lab.machines.values():
        machine.api_object = Mock()
        machine.api_object.reload = Mock()
    return lab


def _fake_get_archive(path, content=b"data"):
    """Emulate container.get_archive: return (tar bytes iterator, stat)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(os.path.basename(path))
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return [buf.getvalue()], {"name": os.path.basename(path), "size": len(content), "mode": 0o644}


def _configure_diff(lab):
    for machine in lab.machines.values():
        machine.api_object.diff.return_value = [
            {"Path": "/root", "Kind": 0},            # directory, not a leaf -> skipped
            {"Path": "/root/NEWFILE", "Kind": 1},    # added file -> captured
            {"Path": "/etc/hosts", "Kind": 0},       # excluded (Docker-managed)
            {"Path": "/oldfile", "Kind": 2},         # deletion
        ]
        machine.api_object.get_archive.side_effect = lambda path: _fake_get_archive(path)


#
# save_lab: validation
#
def test_save_lab_invocation_error_no_target(docker_manager):
    with pytest.raises(InvocationError):
        docker_manager.save_lab("out.tar")


def test_save_lab_invocation_error_selected_and_excluded(docker_manager, running_lab):
    with pytest.raises(InvocationError):
        docker_manager.save_lab("out.tar", lab=running_lab,
                                selected_machines={"pc1"}, excluded_machines={"pc2"})


def test_save_lab_no_devices_raises(docker_manager, running_lab, tmp_path):
    docker_manager.update_lab_from_api = Mock()
    with pytest.raises(MachineNotFoundError):
        docker_manager.save_lab(str(tmp_path / "s.tar"), lab=running_lab,
                                excluded_machines={"pc1", "pc2"})


#
# save_lab: full-image mode (filesystem_diff=False)
#
def test_save_lab_full_writes_archive(docker_manager, running_lab, tmp_path):
    docker_manager.update_lab_from_api = Mock()
    docker_manager.docker_image = Mock()
    docker_manager.docker_image.commit_container.side_effect = \
        lambda container, repository, tag: f"{repository}:{tag}"
    docker_manager.docker_image.save_image_to_tar.return_value = [b"FAKE_IMG"]

    archive = str(tmp_path / "scenario.tar")
    docker_manager.save_lab(archive, lab=running_lab, filesystem_diff=False)

    assert docker_manager.docker_image.commit_container.call_count == 2
    assert docker_manager.docker_image.remove_image.call_count == 2

    with tarfile.open(archive, "r") as tar:
        names = tar.getnames()
        assert "manifest.json" in names
        assert "images/pc1.tar" in names
        assert "images/pc2.tar" in names
        manifest = json.loads(tar.extractfile("manifest.json").read().decode())
        assert manifest["save_mode"] == "full"
        pc1 = next(m for m in manifest["machines"] if m["name"] == "pc1")
        assert pc1["meta"]["image"].startswith("kathara_save_")
        assert pc1["original_image"] == "kathara/test1"


def test_save_lab_full_selected_machines(docker_manager, running_lab, tmp_path):
    docker_manager.update_lab_from_api = Mock()
    docker_manager.docker_image = Mock()
    docker_manager.docker_image.commit_container.side_effect = \
        lambda container, repository, tag: f"{repository}:{tag}"
    docker_manager.docker_image.save_image_to_tar.return_value = [b"FAKE_IMG"]

    archive = str(tmp_path / "scenario.tar")
    docker_manager.save_lab(archive, lab=running_lab, selected_machines={"pc1"}, filesystem_diff=False)

    with tarfile.open(archive, "r") as tar:
        manifest = json.loads(tar.extractfile("manifest.json").read().decode())
        assert {m["name"] for m in manifest["machines"]} == {"pc1"}
        assert "images/pc1.tar" in tar.getnames()
        assert "images/pc2.tar" not in tar.getnames()


#
# save_lab: filesystem-diff mode (default)
#
def test_save_lab_diff_writes_archive(docker_manager, running_lab, tmp_path):
    docker_manager.update_lab_from_api = Mock()
    _configure_diff(running_lab)

    archive = str(tmp_path / "scenario.tar")
    docker_manager.save_lab(archive, lab=running_lab)  # diff is the default

    with tarfile.open(archive, "r") as tar:
        names = tar.getnames()
        assert "images/pc1.diff.tar" in names
        assert "images/pc2.diff.tar" in names
        assert "images/pc1.tar" not in names  # no full images in diff mode

        manifest = json.loads(tar.extractfile("manifest.json").read().decode())
        assert manifest["save_mode"] == "diff"
        pc1 = next(m for m in manifest["machines"] if m["name"] == "pc1")
        assert pc1["original_image"] == "kathara/test1"
        assert pc1["deletions"] == ["/oldfile"]

        diff_bytes = tar.extractfile("images/pc1.diff.tar").read()

    with tarfile.open(fileobj=io.BytesIO(diff_bytes)) as diff_tar:
        diff_names = diff_tar.getnames()
        assert "root/NEWFILE" in diff_names          # added file captured with full path
        assert "etc/hosts" not in diff_names          # excluded path not captured


def test_save_lab_diff_of_a_restored_device_is_relative_to_its_real_image(docker_manager, running_lab, tmp_path):
    """A device restored from a save that had deleted files runs an image built by restore_lab:
    the next save names the image of the device itself and carries the deletions over."""
    docker_manager.update_lab_from_api = Mock()
    _configure_diff(running_lab)
    running_lab.get_machine("pc1").add_meta("image", "kathara_save_h:pc1")
    running_lab.get_machine("pc1").api_object.image.labels = {
        SAVE_BASE_LABEL: "kathara/real", SAVE_DELETIONS_LABEL: encode_deletions(["/gone/before", "/oldfile"]),
    }

    archive = str(tmp_path / "scenario.tar")
    docker_manager.save_lab(archive, lab=running_lab)

    with tarfile.open(archive, "r") as tar:
        manifest = json.loads(tar.extractfile("manifest.json").read().decode())
    pc1 = next(m for m in manifest["machines"] if m["name"] == "pc1")
    pc2 = next(m for m in manifest["machines"] if m["name"] == "pc2")
    assert pc1["original_image"] == "kathara/real"
    assert pc1["deletions"] == ["/gone/before", "/oldfile"]
    assert pc2["original_image"] == "kathara/test2"
    assert pc2["deletions"] == ["/oldfile"]


def test_save_lab_diff_refuses_a_device_whose_deletions_were_not_kept(docker_manager, running_lab, tmp_path):
    docker_manager.update_lab_from_api = Mock()
    _configure_diff(running_lab)
    running_lab.get_machine("pc1").api_object.image.labels = {
        SAVE_BASE_LABEL: "kathara/real", SAVE_DELETIONS_LABEL: DELETIONS_OVERFLOW,
    }

    with pytest.raises(SaveLabError):
        docker_manager.save_lab(str(tmp_path / "scenario.tar"), lab=running_lab)


def test_build_device_diff_spools_a_large_file(docker_manager):
    container = Mock()
    container.diff.return_value = [{"Path": "/root/big", "Kind": 1}]
    content = os.urandom(docker_manager._DIFF_SPOOL_SIZE + 4096)
    tar_bytes = _fake_get_archive("/root/big", content)[0][0]
    # as the Docker API streams it: in chunks
    chunks = [tar_bytes[i:i + 65536] for i in range(0, len(tar_bytes), 65536)]
    container.get_archive.side_effect = lambda path: (iter(chunks), {})

    diff_path, deletions = docker_manager._build_device_diff(container)
    try:
        assert deletions == []
        with tarfile.open(diff_path) as tar:
            assert tar.extractfile("root/big").read() == content
    finally:
        os.remove(diff_path)


def test_build_device_diff_excludes_and_deletes(docker_manager):
    container = Mock()
    container.diff.return_value = [
        {"Path": "/root/keep", "Kind": 1},
        {"Path": "/run/mount", "Kind": 1},      # excluded prefix
        {"Path": "/hosthome", "Kind": 1},       # excluded exact
        {"Path": "/gone", "Kind": 2},           # deletion
        {"Path": "/dev/null", "Kind": 0},       # excluded prefix
    ]
    container.get_archive.side_effect = lambda path: _fake_get_archive(path)

    diff_path, deletions = docker_manager._build_device_diff(container)
    try:
        assert deletions == ["/gone"]
        with tarfile.open(diff_path) as tar:
            names = tar.getnames()
        assert "root/keep" in names
        assert not any(n.startswith("run") or n.startswith("hosthome") or n.startswith("dev") for n in names)
    finally:
        os.remove(diff_path)


#
# restore_lab: full-image mode
#
def _make_full_archive(path, manifest, images, lab_files=None):
    with tarfile.open(path, "w") as tar:
        def add(name, data):
            ti = tarfile.TarInfo(name=name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))

        add("manifest.json", json.dumps(manifest).encode())
        for name, data in images.items():
            add(f"images/{name}.tar", data)
        for name, data in (lab_files or {}).items():
            add(f"lab{name}", data)


def test_restore_lab_full_loads_images_and_deploys(docker_manager, tmp_path):
    manifest = {
        "save_format_version": 1, "save_mode": "full",
        "lab": {"name": "restored", "hash": "somehash", "general_options": {},
                "global_machine_metadata": {}},
        "machines": [{"name": "pc1", "meta": {"image": "kathara_save_x:pc1"},
                      "interfaces": [{"number": 0, "link": "A", "mac_address": None}]}],
        "links": [{"name": "A"}],
    }
    archive = str(tmp_path / "scenario.tar")
    _make_full_archive(archive, manifest, {"pc1": b"FAKE_IMG"}, {"/pc1.startup": b"echo hi\n"})

    docker_manager.docker_image = Mock()
    docker_manager.deploy_lab = Mock()

    lab = docker_manager.restore_lab(archive)

    docker_manager.docker_image.load_images_from_tar.assert_called_once()
    docker_manager.deploy_lab.assert_called_once()
    deployed = docker_manager.deploy_lab.call_args[0][0]
    assert isinstance(deployed, Lab)
    assert deployed.hash == "somehash"
    assert deployed.get_machine("pc1").meta["image"] == "kathara_save_x:pc1"
    assert lab.fs.readbytes("/pc1.startup") == b"echo hi\n"


def test_restore_lab_invalid_archive(docker_manager, tmp_path):
    archive = str(tmp_path / "bad.tar")
    with tarfile.open(archive, "w") as tar:
        data = b"nope"
        ti = tarfile.TarInfo(name="something.txt")
        ti.size = len(data)
        tar.addfile(ti, io.BytesIO(data))

    with pytest.raises(InvocationError):
        docker_manager.restore_lab(archive)


#
# restore_lab: filesystem-diff mode
#
def _make_diff_archive(path, manifest, diffs):
    with tarfile.open(path, "w") as tar:
        def add(name, data):
            ti = tarfile.TarInfo(name=name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))

        add("manifest.json", json.dumps(manifest).encode())
        for name, data in diffs.items():
            add(f"images/{name}.diff.tar", data)


def _diff_manifest(deletions):
    return {
        "save_format_version": 1, "save_mode": "diff",
        "lab": {"name": "restored", "hash": "Some-Hash", "general_options": {},
                "global_machine_metadata": {}},
        "machines": [{"name": "pc1", "meta": {"image": "kathara_save_h:pc1"},
                      "original_image": "kathara/test1", "deletions": deletions,
                      "interfaces": [{"number": 0, "link": "A", "mac_address": None}]}],
        "links": [{"name": "A"}],
    }


def _restore(docker_manager, archive, **kwargs):
    """restore_lab with the image and deploy layers mocked: the lab, and what the deployment saw
    of each device (its image and the content of its saved diff)."""
    seen = {}

    def deploy(lab):
        for name, machine in lab.machines.items():
            diff_path = machine.meta.get(RESTORE_DIFF_META)
            seen[name] = (machine.meta["image"], diff_path, open(diff_path, "rb").read() if diff_path else None)

    docker_manager.docker_image = Mock()
    docker_manager.deploy_lab = Mock(side_effect=deploy)
    return docker_manager.restore_lab(archive, **kwargs), seen


def test_restore_lab_diff_copies_the_saved_files_on_the_base_image(docker_manager, tmp_path):
    archive = str(tmp_path / "scenario.tar")
    _make_diff_archive(archive, _diff_manifest([]), {"pc1": b"DIFFTAR"})

    lab, seen = _restore(docker_manager, archive)

    docker_manager.docker_image.check_from_list.assert_called_once_with({"kathara/test1"})
    docker_manager.docker_image.build_image_without.assert_not_called()
    docker_manager.deploy_lab.assert_called_once()
    # the device runs its own image; its saved files are handed to its creation
    image, diff_path, diff = seen["pc1"]
    assert image == "kathara/test1"
    assert diff == b"DIFFTAR"
    # nothing of the restore is left once the scenario is deployed
    assert not os.path.exists(diff_path)
    assert RESTORE_DIFF_META not in lab.get_machine("pc1").meta


def test_restore_lab_diff_builds_an_image_only_for_the_deletions(docker_manager, tmp_path):
    archive = str(tmp_path / "scenario.tar")
    _make_diff_archive(archive, _diff_manifest(["/x"]), {"pc1": b"DIFFTAR"})

    lab, seen = _restore(docker_manager, archive)

    # the base image without the deleted files, under the hash of the restored scenario
    docker_manager.docker_image.build_image_without.assert_called_once_with(
        "kathara/test1", "kathara_save_somehash:pc1", ["/x"])
    assert seen["pc1"][0] == "kathara_save_somehash:pc1"
    assert seen["pc1"][2] == b"DIFFTAR"
    assert lab.get_machine("pc1").meta["image"] == "kathara_save_somehash:pc1"


def test_restore_lab_diff_cleans_up_when_the_deployment_fails(docker_manager, tmp_path):
    archive = str(tmp_path / "scenario.tar")
    _make_diff_archive(archive, _diff_manifest([]), {"pc1": b"DIFFTAR"})
    paths = []

    def deploy(lab):
        paths.append(lab.get_machine("pc1").meta[RESTORE_DIFF_META])
        raise RuntimeError("boom")

    docker_manager.docker_image = Mock()
    docker_manager.deploy_lab = Mock(side_effect=deploy)
    caller_lab = _caller_lab()
    with pytest.raises(RuntimeError):
        docker_manager.restore_lab(archive, lab=caller_lab)

    assert paths and not os.path.exists(paths[0])
    assert RESTORE_DIFF_META not in caller_lab.get_machine("pc1").meta


#
# restore_lab: caller-provided network scenario (lab=)
#
def _diff_manifest_pc1():
    return {
        "save_format_version": 1, "save_mode": "diff",
        "lab": {"name": "restored", "hash": "somehash", "general_options": {},
                "global_machine_metadata": {}},
        "machines": [{"name": "pc1", "meta": {"image": "kathara_save_h:pc1"},
                      "original_image": "kathara/base", "deletions": [],
                      "interfaces": [{"number": 0, "link": "A", "mac_address": None}]}],
        "links": [{"name": "A"}],
    }


def _caller_lab():
    lab = Lab("caller scenario")
    pc1 = lab.get_or_new_machine("pc1", **{'image': 'kathara/base', 'envs': ['FOO=bar']})
    pc2 = lab.get_or_new_machine("pc2", **{'image': 'kathara/other'})
    lab.connect_machine_to_link(pc1.name, "A")
    lab.connect_machine_to_link(pc2.name, "A")
    lab.shared_path = "/tmp/shared"
    return lab


def test_restore_lab_with_lab_uses_caller_topology_and_saved_images(docker_manager, tmp_path):
    archive = str(tmp_path / "scenario.tar")
    _make_diff_archive(archive, _diff_manifest_pc1(), {"pc1": b"DIFFTAR"})

    docker_manager.docker_image = Mock()
    docker_manager.deploy_lab = Mock()

    caller_lab = _caller_lab()
    lab = docker_manager.restore_lab(archive, lab=caller_lab)

    assert lab is caller_lab
    docker_manager.deploy_lab.assert_called_once()
    assert docker_manager.deploy_lab.call_args[0][0] is caller_lab
    docker_manager.docker_image.build_image_without.assert_not_called()

    # the image the device was saved on, not the one a previous restore had built
    assert caller_lab.get_machine("pc1").meta["image"] == "kathara/base"
    assert caller_lab.get_machine("pc1").meta["envs"] == {"FOO": "bar"}
    assert caller_lab.get_machine("pc2").meta["image"] == "kathara/other"
    assert caller_lab.shared_path == "/tmp/shared"
    assert set(caller_lab.machines.keys()) == {"pc1", "pc2"}


def test_restore_lab_with_lab_ignores_lab_hash(docker_manager, tmp_path):
    archive = str(tmp_path / "scenario.tar")
    _make_diff_archive(archive, _diff_manifest_pc1(), {"pc1": b"DIFFTAR"})

    docker_manager.docker_image = Mock()
    docker_manager.deploy_lab = Mock()

    caller_lab = _caller_lab()
    original_hash = caller_lab.hash
    lab = docker_manager.restore_lab(archive, lab_hash="ignoredhash", lab=caller_lab)

    assert lab.hash == original_hash
    assert lab.hash != "somehash"


def test_restore_lab_with_lab_uploads_lab_files_into_caller_fs(docker_manager, tmp_path):
    manifest = _diff_manifest_pc1()
    manifest["save_mode"] = "full"
    archive = str(tmp_path / "scenario.tar")
    _make_full_archive(archive, manifest, {"pc1": b"FAKE_IMG"}, {"/pc1.startup": b"echo hi\n"})

    docker_manager.docker_image = Mock()
    docker_manager.deploy_lab = Mock()

    caller_lab = _caller_lab()
    lab = docker_manager.restore_lab(archive, lab=caller_lab)

    assert lab is caller_lab
    assert caller_lab.fs.readbytes("/pc1.startup") == b"echo hi\n"
    docker_manager.docker_image.load_images_from_tar.assert_called_once()


#
# undeploy_lab: the images of a restore
#
@mock.patch("src.Kathara.manager.docker.DockerLink.DockerLink.undeploy")
@mock.patch("src.Kathara.manager.docker.DockerMachine.DockerMachine.undeploy")
def test_undeploy_lab_removes_the_images_of_a_restore(mock_undeploy_machine, mock_undeploy_link, docker_manager):
    docker_manager.docker_image = Mock()

    docker_manager.undeploy_lab(lab_hash="Some-Hash", selected_machines={"pc1"})

    docker_manager.docker_image.remove_saved_images.assert_called_once_with(
        "kathara_save_somehash", selected={"pc1"}, excluded=None)


#
# DockerImage: the image of a restored device that had deleted files
#
def test_deletions_label_round_trip():
    paths = ["/etc/b", "/a b/c", "/é", "/etc/b"]
    assert decode_deletions(encode_deletions(paths)) == sorted(set(paths))
    assert decode_deletions("") == []
    with pytest.raises(SaveLabError):
        decode_deletions(DELETIONS_OVERFLOW)
    with pytest.raises(SaveLabError):
        decode_deletions("not a list")


def test_build_image_without_removes_the_listed_paths():
    client = Mock()
    client.images.build.return_value = (Mock(), [])
    deletions = ["/etc/one", "/a 'quoted' $path", "/bad\nname"]

    DockerImage(client).build_image_without("kathara/base", "kathara_save_h:pc1", deletions)

    kwargs = client.images.build.call_args[1]
    assert kwargs["tag"] == "kathara_save_h:pc1" and kwargs["pull"] is False
    assert kwargs["labels"][SAVE_BASE_LABEL] == "kathara/base"
    # a path with a newline cannot be listed: left out of the image and of its label
    assert decode_deletions(kwargs["labels"][SAVE_DELETIONS_LABEL]) == ["/a 'quoted' $path", "/etc/one"]
    with tarfile.open(fileobj=kwargs["fileobj"]) as context:
        dockerfile = context.extractfile("Dockerfile").read().decode()
        listed = context.extractfile(DELETIONS_FILE).read().decode()
    assert dockerfile.startswith("FROM kathara/base\n")
    assert listed == "/etc/one\n/a 'quoted' $path\n"
    # the paths never appear on a Dockerfile line (65535 bytes at most, and a shell to quote for)
    assert "/etc/one" not in dockerfile


def test_build_image_without_too_many_deletions_marks_the_image():
    client = Mock()
    client.images.build.return_value = (Mock(), [])
    # paths that do not compress: the encoded list is longer than a label may be
    deletions = ["/" + os.urandom(48).hex() for _ in range(MAX_DELETIONS_LABEL // 64)]

    DockerImage(client).build_image_without("kathara/base", "kathara_save_h:pc1", deletions)

    assert client.images.build.call_args[1]["labels"][SAVE_DELETIONS_LABEL] == DELETIONS_OVERFLOW


def test_saved_state_of():
    image = Mock()
    image.labels = {SAVE_BASE_LABEL: "kathara/base", SAVE_DELETIONS_LABEL: encode_deletions(["/x"])}
    assert DockerImage.saved_state_of(image) == ("kathara/base", ["/x"])
    image.labels = {"other": "label"}
    assert DockerImage.saved_state_of(image) == (None, [])
    image.labels = None
    assert DockerImage.saved_state_of(image) == (None, [])


def test_remove_saved_images():
    from docker.errors import APIError
    client = Mock()
    mine, other, used = Mock(), Mock(), Mock()
    mine.tags = ["kathara_save_h:pc1", "kathara_save_h:pc2"]
    other.tags = ["kathara_save_hh:pc1"]       # another scenario whose repository starts alike
    used.tags = ["kathara_save_h:pc3"]
    client.images.list.return_value = [mine, other, used]

    def remove(reference):
        if reference == "kathara_save_h:pc3":
            raise APIError("conflict: a container is using the image")

    client.images.remove.side_effect = remove

    DockerImage(client).remove_saved_images("kathara_save_h")
    assert [c[0][0] for c in client.images.remove.call_args_list] == \
        ["kathara_save_h:pc1", "kathara_save_h:pc2", "kathara_save_h:pc3"]

    client.images.remove.reset_mock()
    DockerImage(client).remove_saved_images("kathara_save_h", selected={"pc2", "pc9"})
    assert [c[0][0] for c in client.images.remove.call_args_list] == ["kathara_save_h:pc2"]

    client.images.remove.reset_mock()
    DockerImage(client).remove_saved_images("kathara_save_h", excluded={"pc2"})
    assert [c[0][0] for c in client.images.remove.call_args_list] == ["kathara_save_h:pc1", "kathara_save_h:pc3"]
