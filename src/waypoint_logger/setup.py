from setuptools import find_packages, setup

package_name = 'waypoint_logger'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xiachu',
    maintainer_email='xiachu@andrew.cmu.edu',
    description='Waypoint logging node for simulation, RViz clicked points, and particle-filter odometry.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'waypoint_logger = waypoint_logger.waypoint_logger:main',
        ],
    },
)
