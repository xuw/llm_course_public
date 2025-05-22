import ray
from ray.util import list_named_actors

# 确保 Ray 已经初始化
if not ray.is_initialized():
    ray.init()

# 列出所有命名的 actor
named_actors = list_named_actors(all_namespaces=True)

# 检查是否存在 register_center_actor
actor_name = "register_center_actor"
if actor_name in named_actors:
    print(f"Actor '{actor_name}' is registered in the Ray cluster.")
else:
    print(f"Actor '{actor_name}' is NOT registered in the Ray cluster.")