from setuptools import setup

package_name = "path_follower"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/path_follower.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="wzy",
    maintainer_email="wzy@todo.todo",
    description="路径跟随控制节点",
    license="MIT",
    entry_points={
        "console_scripts": [
            "path_follower_node = path_follower.path_follower_node:main",
        ],
    },
)
