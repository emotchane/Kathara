import io
import json
import sys
import tarfile
from unittest import mock
from unittest.mock import Mock

import pytest

sys.path.insert(0, './')

from src.Kathara.manager.docker.DockerManager import DockerManager
from src.Kathara.model.Lab import Lab
from src.Kathara.exceptions import InvocationError, MachineNotFoundError


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


#
# save_lab
#
def test_save_lab_invocation_error_no_target(docker_manager):
    with pytest.raises(InvocationError):
        docker_manager.save_lab("out.tar")


def test_save_lab_invocation_error_selected_and_excluded(docker_manager, running_lab):
    with pytest.raises(InvocationError):
        docker_manager.save_lab("out.tar", lab=running_lab,
                                selected_machines={"pc1"}, excluded_machines={"pc2"})


def test_save_lab_writes_archive(docker_manager, running_lab, tmp_path):
    docker_manager.update_lab_from_api = Mock()
    docker_manager.docker_image = Mock()
    docker_manager.docker_image.commit_container.side_effect = \
        lambda container, repository, tag: f"{repository}:{tag}"
    docker_manager.docker_image.save_image_to_tar.return_value = [b"FAKE_IMG"]

    archive = str(tmp_path / "scenario.tar")
    docker_manager.save_lab(archive, lab=running_lab)

    # Each running device was committed and its committed image exported.
    assert docker_manager.docker_image.commit_container.call_count == 2
    # Intermediate committed images are cleaned up afterwards.
    assert docker_manager.docker_image.remove_image.call_count == 2

    with tarfile.open(archive, "r") as tar:
        names = tar.getnames()
        assert "manifest.json" in names
        assert "images/pc1.tar" in names
        assert "images/pc2.tar" in names

        manifest = json.loads(tar.extractfile("manifest.json").read().decode())
        assert {m["name"] for m in manifest["machines"]} == {"pc1", "pc2"}
        # Each device points at its committed image and records the original one.
        pc1 = next(m for m in manifest["machines"] if m["name"] == "pc1")
        assert pc1["meta"]["image"].startswith("kathara_save_")
        assert pc1["original_image"] == "kathara/test1"


def test_save_lab_selected_machines(docker_manager, running_lab, tmp_path):
    docker_manager.update_lab_from_api = Mock()
    docker_manager.docker_image = Mock()
    docker_manager.docker_image.commit_container.side_effect = \
        lambda container, repository, tag: f"{repository}:{tag}"
    docker_manager.docker_image.save_image_to_tar.return_value = [b"FAKE_IMG"]

    archive = str(tmp_path / "scenario.tar")
    docker_manager.save_lab(archive, lab=running_lab, selected_machines={"pc1"})

    with tarfile.open(archive, "r") as tar:
        manifest = json.loads(tar.extractfile("manifest.json").read().decode())
        assert {m["name"] for m in manifest["machines"]} == {"pc1"}
        assert "images/pc1.tar" in tar.getnames()
        assert "images/pc2.tar" not in tar.getnames()


def test_save_lab_no_devices_raises(docker_manager, running_lab, tmp_path):
    docker_manager.update_lab_from_api = Mock()
    with pytest.raises(MachineNotFoundError):
        docker_manager.save_lab(str(tmp_path / "s.tar"), lab=running_lab,
                                excluded_machines={"pc1", "pc2"})


#
# restore_lab
#
def _make_archive(path, manifest, images, lab_files=None):
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


def test_restore_lab_loads_images_and_deploys(docker_manager, tmp_path):
    manifest = {
        "save_format_version": 1,
        "lab": {"name": "restored", "hash": "somehash", "general_options": {},
                "global_machine_metadata": {}},
        "machines": [{"name": "pc1", "meta": {"image": "kathara_save_x:pc1"},
                      "interfaces": [{"number": 0, "link": "A", "mac_address": None}]}],
        "links": [{"name": "A"}],
    }
    archive = str(tmp_path / "scenario.tar")
    _make_archive(archive, manifest, {"pc1": b"FAKE_IMG"}, {"/pc1.startup": b"echo hi\n"})

    docker_manager.docker_image = Mock()
    docker_manager.deploy_lab = Mock()

    lab = docker_manager.restore_lab(archive)

    docker_manager.docker_image.load_images_from_tar.assert_called_once()
    docker_manager.deploy_lab.assert_called_once()
    deployed = docker_manager.deploy_lab.call_args[0][0]
    assert isinstance(deployed, Lab)
    assert deployed.hash == "somehash"
    assert set(deployed.machines.keys()) == {"pc1"}
    assert deployed.get_machine("pc1").meta["image"] == "kathara_save_x:pc1"
    # The lab filesystem file has been restored.
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


def test_restore_lab_hash_override(docker_manager, tmp_path):
    manifest = {
        "save_format_version": 1,
        "lab": {"name": "restored", "hash": "somehash", "general_options": {},
                "global_machine_metadata": {}},
        "machines": [{"name": "pc1", "meta": {}, "interfaces": []}],
        "links": [],
    }
    archive = str(tmp_path / "scenario.tar")
    _make_archive(archive, manifest, {"pc1": b"FAKE_IMG"})

    docker_manager.docker_image = Mock()
    docker_manager.deploy_lab = Mock()

    lab = docker_manager.restore_lab(archive, lab_hash="override")
    assert lab.hash == "override"
