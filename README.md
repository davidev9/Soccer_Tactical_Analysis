# Temporal Knowledge Graph Construction from Football Match Videos

## Folder Structure
- `src`: contains the source code for the entire project.

## How to Start the System - Distributed Version

### Environment Variables
Before starting the system components, the following environment variables must be set to provide the required credentials for Redis and ArangoDB:

### Redis Configuration
- `REDIS_HOST`: The Redis host address.
- `REDIS_PORT`: The port number for Redis.
- `REDIS_PASSWORD`: The password for Redis.

### ArangoDB Configuration
- `ARANGO_ENCODEDCA`: The encoded CA certificate for a secure ArangoDB connection.
- `ARANGO_HOST`: The URL of the ArangoDB instance.
- `ARANGO_USERNAME`: The username for ArangoDB.
- `ARANGO_PASSWORD`: The password for ArangoDB.


### Run the RT processor

```bash
python rt_player_tracker.py \
    --video_path "/path/to/video.mp4" \
    --player_model_path "/path/to/player_model.pt" \
    --field_model_path "/path/to/field_model.pt" \
    --output_path "/path/to/output/result.mp4"
```
or run  `RT_Processor` notebook

### Run the Advanced Analysis module (and the backup)

```bash
python advanced_analysis.py
python advanced_analysis_backup.py
```
or run  `Advanced_Analysis` notebook


## How to Start the System - Single Node Version

### Environment Variables
Before starting the system components, the following environment variables must be set to provide the required credentials for Redis and ArangoDB:

### ArangoDB Configuration
- `ARANGO_ENCODEDCA`: The encoded CA certificate for a secure ArangoDB connection.
- `ARANGO_HOST`: The URL of the ArangoDB instance.
- `ARANGO_USERNAME`: The username for ArangoDB.
- `ARANGO_PASSWORD`: The password for ArangoDB.


### Run the Single Node Processor

```bash
python single_node_processor.py \
    --video_path "/path/to/video.mp4" \
    --player_model_path "/path/to/player_model.pt" \
    --field_model_path "/path/to/field_model.pt" \
    --output_path "/path/to/output/result.mp4"
```




