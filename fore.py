# -*- coding: utf-8 -*-
import logging
from scene import *

# 示例使用
original_filename = "try_script_log.log"
new_filename = add_timestamp_to_filename(original_filename)
# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(new_filename),
        logging.StreamHandler()
    ]
)




scenes_generated = dynamic_scene_generation(input_data6)
scenes_generated=add_foreshadowing_and_clues_to_scene_list(scenes_generated)
# 如果需要改变为锁闭式结构
# scenes_generated = reorder_scenes_for_closure_structure(scenes_generated)

# logging.info("All scenes generated and reordered for closure structure.")


for scene in scenes_generated:
    print(scene["description"])
save_scenes_to_json(scenes_generated,'zhengxu6.json')

# scenes_generated = dynamic_scene_generation(input_data4)
# scenes_generated=add_foreshadowing_and_clues_to_scene_list(scenes_generated)
# # 如果需要改变为锁闭式结构
# # scenes_generated = reorder_scenes_for_closure_structure(scenes_generated)

# # logging.info("All scenes generated and reordered for closure structure.")


# for scene in scenes_generated:
#     print(scene["description"])
# save_scenes_to_json(scenes_generated,'6.json')