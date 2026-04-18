@echo off
REM 智能眼镜人脸识别Demo - 一键运行脚本
REM 使用方法: 双击此文件或从命令行运行

cd /d "%~dp0"
echo ============================================
echo 智能眼镜人脸识别 Demo
echo ============================================
echo.

echo 正在激活虚拟环境...
call Scripts\activate.bat

echo.
echo 欢迎使用人脸识别Demo！
echo.
echo 命令说明:
echo   1. register ^<图片路径^> ^<姓名^>  - 注册新人脸
echo   2. recognize ^<图片路径^>          - 识别图片中的人脸
echo   3. list                            - 查看已注册人脸
echo   4. load-dir ^<目录^>              - 批量加载目录下的人脸图片
echo   5. camera                          - 打开摄像头实时识别
echo   6. help                            - 显示帮助
echo.
echo 示例:
echo   register ..\sample_faces\known\Al_Sharpton_0.jpg Al_Sharpton
echo   recognize ..\sample_faces\known\Al_Sharpton_0.jpg
echo   list
echo.
echo 按任意键退出...
pause > nul
