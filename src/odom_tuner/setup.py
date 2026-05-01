from setuptools import find_packages, setup

package_name = 'odom_tuner'

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
    description='Odometry tuning tools for F1TENTH.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'odom_tuner = odom_tuner.odom_tuner:main',
            'odom_noise_relay = odom_tuner.odom_noise_relay:main',
        ],
    },
)
