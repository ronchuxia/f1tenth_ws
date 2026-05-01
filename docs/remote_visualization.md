# Remote Visualization

University Wi-Fi blocks **UDP multicast**, preventing ROS 2 nodes on the same subnet from discovering each other automatically. 

The **CycloneDDS** configuration below uses explicit peer discovery over **UDP unicast**, so ROS 2 nodes can communicate over the university network.

This allows us to use RViz 2 on our laptop to visualize the code running on the jetson.

1. Create a `~/cyclonedds.xml` file on your laptop with the following content:

    ```xml
    <CycloneDDS>
    <Domain>
        <Discovery>
        <Peers>
            <Peer address="team9-desktop"/>
        </Peers>
        <ParticipantIndex>auto</ParticipantIndex>
        <MaxAutoParticipantIndex>100</MaxAutoParticipantIndex>
        </Discovery>
    </Domain>
    </CycloneDDS>
    ```

    - `address`: The hostname or IP address of the jetson.
    - `MaxAutoParticipantIndex`: The maximum index of participants.

2. On your laptop, add these to `~/.bashrc`:

    ```shell
    export ROS_DOMAIN_ID=9
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI=$HOME/cyclonedds.xml
    ```

3. Do the same on the jetson.
