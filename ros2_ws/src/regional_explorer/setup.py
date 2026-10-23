from setuptools import find_packages, setup

package_name = "regional_explorer"

setup(
    name=package_name,
    version="0.1.0",
    # 用 find_packages 而不是写死单包：这样 regional_explorer/legacy（随包携带的
    # 算法副本，含 core/ 与 real_work/）在**非 symlink 安装**时也会被装进
    # site-packages，保证只拷 ros2_ws 就能跑仿真/真机。
    packages=find_packages(exclude=["test"]),
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
