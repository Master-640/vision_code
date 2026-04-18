import os
import sys

demo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, demo_dir)

from main import SmartGlassesDemo

known_dir = os.path.join(demo_dir, 'sample_faces', 'known')
db_path = os.path.join(demo_dir, 'data', 'face_quick.json')

# 清空旧数据库
if os.path.exists(db_path):
    os.remove(db_path)

print("="*50)
print("  智能眼镜人脸识别 Demo")
print("="*50)

d = SmartGlassesDemo(db_path)

# 测试哪些人
targets = ['Al_Sharpton', 'George_W_Bush', 'Bill_Gates', 'Colin_Powell', 'Arnold_Schwarzenegger']
registered = []

print("\n[1] 注册人脸...")
for target in targets:
    for f in sorted(os.listdir(known_dir)):
        if f.startswith(target + '_') and f.endswith('.jpg'):
            path = os.path.join(known_dir, f)
            if d.register_face(path, target):
                registered.append((target, f))
                print(f"  注册成功: {target} ({f})")
                break
            else:
                print(f"  注册失败: {target} ({f}) - 未检测到人脸")

if len(registered) == 0:
    print("警告: 没有成功注册任何人脸，尝试其他图片...")
    for f in sorted(os.listdir(known_dir))[:20]:
        if f.endswith('.jpg'):
            path = os.path.join(known_dir, f)
            name = os.path.splitext(f)[0].rsplit('_', 1)[0]
            if name not in [r[0] for r in registered]:
                if d.register_face(path, name):
                    registered.append((name, f))
                    print(f"  注册成功: {name} ({f})")
                if len(registered) >= 3:
                    break

print(f"\n共注册: {len(registered)} 人")

# 识别测试
print("\n[2] 识别测试...")
d.reload_faces()
print(f"训练集: {len(d.recognizer.train_images)} 张图片")

success = 0
for name, reg_file in registered:
    test_path = os.path.join(known_dir, reg_file)
    results = d.recognize_from_image(test_path)
    if results:
        pred_name = results[0]['name']
        conf = results[0]['confidence']
        match = "正确" if pred_name == name else f"错误(应为{name})"
        print(f"  {name} -> {pred_name} (置信度: {conf:.0f}) {match}")
        if pred_name == name:
            success += 1
    else:
        print(f"  {name} -> 无识别结果")

print(f"\n识别成功率: {success}/{len(registered)}")
print("\nDemo 运行完成!")
