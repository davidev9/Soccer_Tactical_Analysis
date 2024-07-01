from redis import Redis
from PIL import Image
import base64
from io import BytesIO
import json
import matplotlib.pyplot as plt
import os
class Keyframe:
    def __init__(self, image, video_name, minute, second, tracker_data):
        self.image = image
        self.video_name = video_name
        self.minute = minute
        self.second = second
        self.tracker_data = tracker_data

    def to_redis_format(self) -> bytes:
        image_bytes = self.serialize_image()
        data = {
            'video_name': self.video_name,
            'minute': self.minute,
            'second': self.second,
            'tracker_data': self.tracker_data,
            'image': image_bytes
        }
        return json.dumps(data).encode()

    @classmethod
    def from_redis_format(cls, serialized_data: bytes):
        try:
            data = json.loads(serialized_data)
            image = cls.deserialize_image(data['image'])
            return cls(image, data['video_name'], data['minute'], data['second'], data['tracker_data'])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error decoding data: {e}")
            print(f"Serialized data causing error: {serialized_data}")
            return None


    def serialize_image(self) -> str:
        with BytesIO() as output:
            self.image.save(output, format='PNG')
            image_bytes = output.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    @staticmethod
    def deserialize_image(encoded_image: str) -> Image:
        image_bytes = base64.b64decode(encoded_image)
        return Image.open(BytesIO(image_bytes))

class KeyframeQueue:
    def __init__(self, redis_conn: Redis):
        if not isinstance(redis_conn, Redis):
            raise ValueError("redis_conn must be a valid Redis connection!")
        self._r = redis_conn

    def list_len(self) -> int:
        return self._r.llen("keyframes")
    
    def BU_list_len(self) -> int:
        return self._r.llen("keyframes_backup")



    def push_keyframe(self, keyframe: Keyframe) -> int:
        serialized_keyframe = keyframe.to_redis_format()
        return self._r.lpush("keyframes", serialized_keyframe)

    def pop_keyframe(self):
        serialized_keyframe = self._r.rpoplpush("keyframes", "keyframes_backup")
        if not serialized_keyframe:
            return (None, None)

        keyframe = Keyframe.from_redis_format(serialized_keyframe)
        return serialized_keyframe, keyframe

    def peek_keyframe_backup(self):
        serialized_keyframe = self._r.lrange("keyframes_backup", -1, -1)
        if not serialized_keyframe:
            return (None, None)


        keyframe = Keyframe.from_redis_format(serialized_keyframe[0])
        return serialized_keyframe, keyframe

    def delete_backup(self):
        self._r.delete("keyframes_backup")

    def delete_main(self):
        self._r.delete("keyframes")


    def del_keyframe_from_backup(self, keyframe_ref: bytes):
        if keyframe_ref:
            self._r.lrem("keyframes_backup", 1, keyframe_ref)


def main():
    redis_conn = Redis(
        host=os.getenv("REDIS_HOST"),
        port=os.getenv("REDIS_PORT"),
        password=os.getenv("REDIS_PASSWORD"),
        decode_responses=True
    )
    
    # Initializing the keyframe queue
    kf_queue = KeyframeQueue(redis_conn)
    ciao=kf_queue.list_len()
    print("coda princ",ciao)
    keyframe_ref, retrieved_keyframe = kf_queue.pop_keyframe()
    img_frame=retrieved_keyframe.image
    plt.imshow(img_frame)
    plt.axis('off')  # Nascondi gli assi
    plt.show()
    ciao=kf_queue.BU_list_len()
    print("coda back prima",ciao)
    kf_queue.del_keyframe_from_backup(keyframe_ref)
    ciao=kf_queue.BU_list_len()
    print("coda back dopo",ciao)



if __name__ == "__main__":
    main()
