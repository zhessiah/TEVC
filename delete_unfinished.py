import os
import shutil

ROOT_DIR = "results/tb_logs/EA/adversarial_training"
VERY_SMALL_STEP = 5_000   # 只删 <= 5000 step 的

to_delete = []

for map_name in os.listdir(ROOT_DIR):
    map_path = os.path.join(ROOT_DIR, map_name)
    if not os.path.isdir(map_path):
        continue

    for run_name in os.listdir(map_path):
        run_path = os.path.join(map_path, run_name)
        models_path = os.path.join(run_path, "models")

        if not os.path.isdir(models_path):
            to_delete.append((run_path, -1))
            continue
        steps = [int(d) for d in os.listdir(models_path) if d.isdigit()]
        if not steps:
            continue

        max_step = max(steps)

        if max_step <= VERY_SMALL_STEP:
            to_delete.append((run_path, max_step))

# 再确认一遍
print("⚠️ 即将删除的 run（极小 step）：\n")
for p, s in to_delete:
    print(f"{p}  --> max_step={s}")

print(f"\n总计：{len(to_delete)} 个")

# ===== 真删（你确认后再打开）=====
# for p, _ in to_delete:
#     shutil.rmtree(p)
