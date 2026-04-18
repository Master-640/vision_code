import os
import sys

demo_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(demo_dir, 'venv', 'Scripts', 'python.exe')
sys.path.insert(0, demo_dir)

if len(sys.argv) < 2:
    print("""
============================================
智能眼镜人脸识别 Demo
============================================

用法: python demo.py <命令> [参数]

命令:
  register <图片路径> <姓名>  - 注册新人脸
  recognize <图片路径>        - 识别图片中的人脸
  list                       - 查看已注册人脸
  load-dir <目录>           - 批量加载目录下的人脸图片
  help                      - 显示帮助

示例:
  python demo.py register sample_faces/known/Al_Sharpton_0.jpg Al_Sharpton
  python demo.py recognize sample_faces/known/Al_Sharpton_0.jpg
  python demo.py list
""")
    sys.exit(1)

from main import SmartGlassesDemo

db_path = os.path.join(demo_dir, 'data', 'face_demo.json')
d = SmartGlassesDemo(db_path)
d.reload_faces()

command = sys.argv[1].lower()

if command == 'register' and len(sys.argv) >= 4:
    image_path = sys.argv[2]
    name = sys.argv[3]
    if not os.path.isabs(image_path):
        image_path = os.path.join(demo_dir, image_path)
    if d.register_face(image_path, name):
        print(f"成功注册: {name}")
    else:
        print(f"注册失败，请检查图片路径是否正确")

elif command == 'recognize' and len(sys.argv) >= 3:
    image_path = sys.argv[2]
    if not os.path.isabs(image_path):
        image_path = os.path.join(demo_dir, image_path)
    results = d.recognize_from_image(image_path)
    print(f"\n识别结果:")
    for i, r in enumerate(results):
        print(f"  人脸{i+1}: {r['name']} (置信度: {r['confidence']:.1f})")

elif command == 'list':
    names = d.list_registered()

elif command == 'load-dir' and len(sys.argv) >= 3:
    dir_path = sys.argv[2]
    if not os.path.isabs(dir_path):
        dir_path = os.path.join(demo_dir, dir_path)
    count = 0
    for f in os.listdir(dir_path):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            name = os.path.splitext(f)[0]
            if d.register_face(os.path.join(dir_path, f), name):
                count += 1
    print(f"成功注册 {count} 张人脸")

elif command == 'help':
    print("""
============================================
智能眼镜人脸识别 Demo
============================================

命令:
  register <图片路径> <姓名>  - 注册新人脸
  recognize <图片路径>        - 识别图片中的人脸
  list                       - 查看已注册人脸
  load-dir <目录>           - 批量加载目录下的人脸图片
  help                      - 显示帮助

示例:
  python demo.py register sample_faces/known/Al_Sharpton_0.jpg Al_Sharpton
  python demo.py recognize sample_faces/known/Al_Sharpton_0.jpg
""")
else:
    print(f"未知命令: {command}")
    print("使用 'python demo.py help' 查看帮助")
