from setuptools import setup

package_name = "regional_explorer"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/frontier_explorer.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="wzy",
    maintainer_email="wzy@todo.todo",
    description="区域级几何启发式二维栅格自主探索节点",
    license="MIT",
    entry_points={
        "console_scripts": [
            "frontier_explorer_node = regional_explorer.frontier_explorer_node:main",
        ],
    },
)
