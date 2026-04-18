import os
import sys
import time
import json
import cv2
import numpy as np

demo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, demo_dir)

from face_recognizer import FaceRecognizer

known_dir = os.path.join(demo_dir, 'sample_faces', 'known')
report_dir = os.path.join(os.path.dirname(demo_dir), 'report')
os.makedirs(report_dir, exist_ok=True)

def select_candidates(min_images=6):
    """选择有足够图片的人员"""
    person_images = {}
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for f in sorted(os.listdir(known_dir)):
        if not f.endswith('.jpg'):
            continue
        name = os.path.splitext(f)[0].rsplit('_', 1)[0]
        if name not in person_images:
            person_images[name] = []
        person_images[name].append(f)
    
    candidates = []
    for name, imgs in sorted(person_images.items(), key=lambda x: -len(x[1])):
        if len(imgs) >= min_images:
            candidates.append((name, imgs))
        if len(candidates) >= 30:
            break
    
    return candidates

def run_30_person_demo():
    print("="*70)
    print("  智能眼镜人脸识别 Demo - 30人测试")
    print("="*70)
    
    candidates = select_candidates(min_images=6)
    print(f"\n选择人员: {len(candidates)}人")
    for name, imgs in candidates[:10]:
        print(f"  {name}: {len(imgs)}张图片")
    if len(candidates) > 10:
        print(f"  ... 及其他{len(candidates)-10}人")
    
    recognizer = FaceRecognizer()
    registered = []
    
    print("\n[1] 注册人脸（每人5张训练，2张测试）...")
    for name, images in candidates:
        train_files = []
        test_files = []
        
        for img in images:
            path = os.path.join(known_dir, img)
            gray = cv2.imread(path, 0)
            faces = recognizer.face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))
            if len(faces) > 0:
                if len(train_files) < 4:
                    if recognizer.load_known_face(path, name):
                        train_files.append(img)
                elif len(test_files) < 2:
                    test_files.append(img)
        
        if len(train_files) >= 3:
            registered.append({
                'name': name, 
                'train_files': train_files, 
                'test_files': test_files
            })
            print(f"  OK {name}: 训练{len(train_files)}张, 测试{len(test_files)}张")
        else:
            print(f"  SKIP {name}: 训练图片不足({len(train_files)}张)，跳过")
    
    recognizer.train()
    print(f"\n注册完成: {len(registered)}人, 训练图片:{len(recognizer.train_images)}张")
    
    print("\n[2] 识别测试...")
    test_results = []
    total_tests = 0
    success_count = 0
    fail_count = 0
    unknown_count = 0
    
    for person in registered:
        name = person['name']
        test_files = person['test_files']
        person_results = {'name': name, 'train_count': len(person['train_files']), 'tests': []}
        
        for test_file in test_files:
            test_path = os.path.join(known_dir, test_file)
            total_tests += 1
            
            results = recognizer.recognize_all_faces(test_path)
            if results:
                pred_name, conf, _ = results[0]
                match = pred_name == name
                
                # Handle infinite confidence (no match within threshold)
                if conf == float('inf') or conf > 1000:
                    conf_display = "无匹配"
                    conf_numeric = None
                else:
                    conf_display = f"{conf:.0f}"
                    conf_numeric = conf
                
                if match and conf_numeric is not None:
                    success_count += 1
                    status = "OK"
                else:
                    fail_count += 1
                    status = f"ERR(应为{name})"
                
                print(f"  {name}: {test_file} -> {pred_name} (距离:{conf_display}) {status}")
                person_results['tests'].append({
                    'test_file': test_file, 
                    'predicted': pred_name,
                    'confidence': conf_numeric, 
                    'match': match
                })
            else:
                unknown_count += 1
                print(f"  {name}: {test_file} -> 未检测到人脸")
                person_results['tests'].append({
                    'test_file': test_file, 
                    'predicted': 'Unknown',
                    'confidence': None, 
                    'match': False
                })
        test_results.append(person_results)
    
    accuracy_rate = (success_count / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"测试结果汇总:")
    print(f"  注册人数: {len(registered)}")
    print(f"  训练图片数: {len(recognizer.train_images)}")
    print(f"  测试图片数: {total_tests}")
    print(f"  识别正确: {success_count}")
    print(f"  识别错误: {fail_count}")
    print(f"  未识别: {unknown_count}")
    print(f"  准确率: {accuracy_rate:.1f}%")
    print(f"{'='*70}")
    
    ts = time.strftime('%Y%m%d_%H%M%S')
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'algorithm': 'LBPH (Local Binary Patterns Histograms)',
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
        
        f.write("## 关于置信度的说明\n")
        f.write("LBPH算法返回的是**卡方距离**（chi-square distance），值越小表示匹配度越高：\n")
        f.write("- **0**: 完美匹配（测试图片与训练图片完全相同）\n")
        f.write("- **< 50**: 高置信度匹配\n")
        f.write("- **50-80**: 中等置信度\n")
        f.write("- **> 80**: 低置信度\n")
        f.write("- **无匹配**: 距离超过阈值（>80），无法识别\n\n")
        f.write("LBPH算法返回\"无匹配\"时，OpenCV会返回float('inf')，表示不匹配任何已知人员。\n\n")
        
        f.write("## 测试结果汇总\n\n")
        f.write(f"| 指标 | 数值 |\n|------|------|\n")
        f.write(f"| 注册人数 | {report['registered_count']} |\n")
        f.write(f"| 训练图片总数 | {len(recognizer.train_images)} |\n")
        f.write(f"| 测试图片数 | {report['total_tests']} |\n")
        f.write(f"| 识别正确 | {report['success_count']} |\n")
        f.write(f"| 识别错误 | {report['fail_count']} |\n")
        f.write(f"| 未识别 | {report['unknown_count']} |\n")
        f.write(f"| **准确率** | **{report['accuracy_rate']}** |\n\n")
        
        f.write("## 详细结果\n\n")
        f.write("| 姓名 | 训练图片数 | 测试图片 | 识别结果 | 距离值 | 状态 |\n")
        f.write("|------|----------|----------|----------|--------|------|\n")
        for person in test_results:
            for t in person['tests']:
                status = "OK正确" if t['match'] else ("X未识别" if t['predicted']=='Unknown' else f"X错误(应为{person['name']})")
                conf = f"{t['confidence']:.0f}" if t['confidence'] is not None else "无匹配"
                f.write(f"| {person['name']} | {person['train_count']} | {t['test_file']} | {t['predicted']} | {conf} | {status} |\n")
        
        f.write("\n## 结论\n\n")
        f.write(f"本次Demo使用30人进行测试，准确率为{accuracy_rate:.1f}%。\n")
        f.write("LBPH作为传统人脸识别算法，在以下场景表现有限：\n")
        f.write("1. 表情、光照、角度变化较大时\n")
        f.write("2. 图片分辨率较低时（LFW数据集为125x94像素）\n")
        f.write("3. 人员数量较多时（30人规模）\n\n")
        f.write("建议：如需更高准确率，可考虑使用基于深度学习的人脸识别算法（如FaceNet、ArcFace等）。\n")
        f.write("\n---\n*报告由系统自动生成*\n")
    
    print(f"\n报告已生成: {md_path}")
    print("\nDemo 运行完成!")

if __name__ == "__main__":
    run_30_person_demo()
