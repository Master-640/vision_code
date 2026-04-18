import os
import sys
import time
import json
import cv2

demo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, demo_dir)

from main import SmartGlassesDemo

known_dir = os.path.join(demo_dir, 'sample_faces', 'known')
report_dir = os.path.join(os.path.dirname(demo_dir), 'report')
os.makedirs(report_dir, exist_ok=True)

print("="*60)
print("  智能眼镜人脸识别 Demo - 10人测试")
print("="*60)

# 筛选人员
person_images = {}
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
for f in sorted(os.listdir(known_dir)):
    if not f.endswith('.jpg'):
        continue
    name = os.path.splitext(f)[0].rsplit('_', 1)[0]
    if name not in person_images:
        person_images[name] = []
    person_images[name].append(f)

# 选10个人
candidates = []
for name, imgs in sorted(person_images.items(), key=lambda x: -len(x[1])):
    if len(imgs) >= 6:
        candidates.append((name, imgs))
    if len(candidates) >= 10:
        break

print(f"\n选择人员: {len(candidates)}人")

# 直接使用FaceRecognizer，不用数据库
from face_recognizer import FaceRecognizer
recognizer = FaceRecognizer()

registered = []
print("\n[1] 注册人脸（每人4张训练）...")
for name, images in candidates:
    train_files = []
    test_files = []
    for img in images:
        path = os.path.join(known_dir, img)
        gray = cv2.imread(path, 0)
        faces = cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))
        if len(faces) > 0:
            if len(train_files) < 4:
                if recognizer.load_known_face(path, name):
                    train_files.append(img)
            else:
                test_files.append(img)
    if len(train_files) > 0:
        registered.append({'name': name, 'train_files': train_files, 'test_files': test_files[:2]})
        print(f"  注册: {name} (训练:{len(train_files)}张, 待测:{len(test_files[:2])}张)")

recognizer.train()
print(f"\n注册完成: {len(registered)} 人, 训练图片:{len(recognizer.train_images)}张")

# 识别测试
print("\n[2] 识别测试...")
test_results = []
total_tests = 0
success_count = 0
fail_count = 0
unknown_count = 0

for person in registered:
    name = person['name']
    test_files = person['test_files']
    person_results = {'name': name, 'train_files': person['train_files'], 'tests': []}

    for test_file in test_files:
        test_path = os.path.join(known_dir, test_file)
        total_tests += 1
        results = recognizer.recognize_all_faces(test_path)
        if results:
            pred_name, conf, _ = results[0]
            acc = 100.0 if conf == 0 else max(0, (1 - conf/150)*100)
            match = pred_name == name
            if match:
                success_count += 1
                status = "正确"
            else:
                fail_count += 1
                status = f"错误(应为{name})"
            print(f"  {name}: {test_file} -> {pred_name} (准确率: {acc:.1f}%) {status}")
            person_results['tests'].append({
                'test_file': test_file, 'predicted': pred_name,
                'confidence': conf, 'accuracy': acc, 'match': match
            })
        else:
            unknown_count += 1
            print(f"  {name}: {test_file} -> 无识别结果")
            person_results['tests'].append({
                'test_file': test_file, 'predicted': 'Unknown',
                'confidence': None, 'accuracy': None, 'match': False
            })
    test_results.append(person_results)

accuracy_rate = (success_count / total_tests * 100) if total_tests > 0 else 0

print(f"\n{'='*60}")
print(f"测试结果汇总:")
print(f"  注册人数: {len(registered)}")
print(f"  训练图片数: {len(recognizer.train_images)}")
print(f"  测试图片数: {total_tests}")
print(f"  识别正确: {success_count}")
print(f"  识别错误: {fail_count}")
print(f"  未识别: {unknown_count}")
print(f"  准确率: {accuracy_rate:.1f}%")
print(f"{'='*60}")

# 生成报告
ts = time.strftime('%Y%m%d_%H%M%S')
report = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'algorithm': 'LBPH',
    'detector': 'Haar Cascade',
    'registered_count': len(registered),
    'total_tests': total_tests,
    'success_count': success_count,
    'fail_count': fail_count,
    'unknown_count': unknown_count,
    'accuracy_rate': f"{accuracy_rate:.1f}%",
    'detailed_results': test_results
}

report_path = os.path.join(report_dir, f"demo_report_{ts}.json")
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

md_path = os.path.join(report_dir, f"demo_report_{ts}.md")
with open(md_path, 'w', encoding='utf-8') as f:
    f.write("# 智能眼镜人脸识别 Demo 测试报告\n\n")
    f.write(f"## 测试时间\n{report['timestamp']}\n\n")
    f.write("## 算法说明\n")
    f.write(f"- 人脸识别: {report['algorithm']}\n")
    f.write(f"- 人脸检测: {report['detector']}\n\n")
    f.write("## 测试结果汇总\n\n")
    f.write(f"| 指标 | 数值 |\n|------|------|\n")
    f.write(f"| 注册人数 | {report['registered_count']} |\n")
    f.write(f"| 测试图片数 | {report['total_tests']} |\n")
    f.write(f"| 识别正确 | {report['success_count']} |\n")
    f.write(f"| 识别错误 | {report['fail_count']} |\n")
    f.write(f"| 未识别 | {report['unknown_count']} |\n")
    f.write(f"| **准确率** | **{report['accuracy_rate']}** |\n\n")
    f.write("## 详细结果\n\n")
    f.write("| 姓名 | 测试图片 | 识别结果 | 准确率 | 状态 |\n|------|----------|----------|--------|------|\n")
    for person in test_results:
        for t in person['tests']:
            status = "正确" if t['match'] else ("未识别" if t['predicted']=='Unknown' else "错误")
            acc = f"{t['accuracy']:.1f}%" if t['accuracy'] else "-"
            f.write(f"| {person['name']} | {t['test_file']} | {t['predicted']} | {acc} | {status} |\n")
    f.write("\n---\n*报告由系统自动生成*\n")

print(f"\n报告: {md_path}")
print("\nDemo 运行完成!")
