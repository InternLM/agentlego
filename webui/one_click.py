import os
import platform
import re
import shutil
import signal
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

script_dir = Path(__file__).absolute().parent
conda_env_path = os.path.join(script_dir, 'installer_files', 'env')
agentlego_root = script_dir.parent

# Remove the '# ' from the following lines as needed for your AMD GPU on Linux
# os.environ["ROCM_PATH"] = '/opt/rocm'
# os.environ["HSA_OVERRIDE_GFX_VERSION"] = '10.3.0'
# os.environ["HCC_AMDGPU_TARGET"] = 'gfx1030'

signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))


def is_linux():
    return sys.platform.startswith('linux')


def is_windows():
    return sys.platform.startswith('win')


def is_macos():
    return sys.platform.startswith('darwin')


def is_x86_64():
    return platform.machine() == 'x86_64'


def cpu_has_avx2():
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        if 'avx2' in info['flags']:
            return True
        else:
            return False
    except Exception:
        return True


def cpu_has_amx():
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        if 'amx' in info['flags']:
            return True
        else:
            return False
    except Exception:
        return True


def digit_version(version_str: str):
    pattern = r'(?P<major>\d+)\.?(?P<minor>\d+)?\.?(?P<patch>\d+)?'
    version = re.match(pattern, version_str)
    assert version is not None, f'failed to parse version {version_str}'
    return tuple(int(i) if i is not None else 0 for i in version.groups())


def get_version(package: str):
    try:
        dist = distribution(package)
        return digit_version(dist.version)
    except PackageNotFoundError:
        return None


def check_env():
    # If we have access to conda, we are probably in an environment
    conda_exist = run_cmd('conda', capture_output=True).returncode == 0
    if not conda_exist:
        print('Conda is not installed. Exiting...')
        sys.exit(1)

    # Ensure this is a new environment and not the base environment
    if os.environ['CONDA_DEFAULT_ENV'] == 'base':
        print('Create an environment for this project and activate it. Exiting...')
        sys.exit(1)


def clear_cache():
    run_cmd('conda clean -a -y')
    run_cmd('python -m pip cache purge')


def print_big_message(message):
    message = message.strip()
    lines = message.split('\n')
    print('\n\n*******************************************************************')
    for line in lines:
        if line.strip() != '':
            print('*', line)

    print('*******************************************************************\n\n')


def run_cmd(cmd, assert_success=False, capture_output=False, env=None):
    # Run shell commands
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, env=env)

    # Assert the command ran successfully
    if assert_success and result.returncode != 0:
        print("Command '" + cmd + "' failed with exit status code '" +
              str(result.returncode) +
              "'.\n\nExiting now.\nTry running the script again.")
        sys.exit(1)

    return result


def install_agentlego():
    agentlego = get_version('agentlego')
    if agentlego is None or not Path(
            distribution('agentlego').locate_file('.')).samefile(agentlego_root):
        print('Installing AgentLego')
        run_cmd(f'python -m pip install -e {agentlego_root}', assert_success=True)


def install_demo_dependencies():
    gradio = get_version('gradio')
    install = []
    if gradio is None or gradio < digit_version('4.13.0'):
        install.append("'gradio>=4.13.0'")

    if get_version('langchain') is None:
        install.append('langchain')
    if get_version('langchain-openai') is None:
        install.append('langchain-openai')
    if get_version('markdown') is None:
        install.append('markdown')

    lagent = get_version('lagent')
    if lagent is None or lagent < digit_version('0.2.0'):
        install.append("'git+https://github.com/mzr1996/lagent@lite'")

    if install:
        run_cmd('python -m pip install ' + ' '.join(install), assert_success=True)

if __name__ == '__main__':
    # Verifies we are in a conda environment
    check_env()
    os.chdir(script_dir)
    if not (script_dir / 'tool_config.yml').exists():
        shutil.copy(script_dir / 'tool_config.yml.example',
                    script_dir / 'tool_config.yml')
    if not (script_dir / 'agent_config.yml').exists():
        shutil.copy(script_dir / 'agent_config.yml.example',
                    script_dir / 'agent_config.yml')

    # Install the current version agentlego
    install_agentlego()

    # Install gradio
    install_demo_dependencies()

    # Install main tools dependencies
    # install_tool_dependencies()

    # Launch the gradio
    run_cmd(f'python app.py ' + ' '.join(sys.argv[1:]))
