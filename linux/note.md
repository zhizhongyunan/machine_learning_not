# Linux学习以及问题笔记

## 命令使用

1. 使用vim编辑文件~/.bashrc 或者~/.profile进行环境变量的修改，添加语句export PATH=$PATH:/path，path替换为可执行文件位置
2. 移动文件命令

   1. ~~~bash
      mv file.txt /home/yanzhengqi/Documents
      mv oldname.txt newname.txt
      mv -i ..."交互式移动"
      ~~~
   2. 主机与虚拟机传输文件
      1. ~~~bash
         scp /path/local/project/your_file username@ip:/home/username/path
         ~~~
3. 多终端执行命令

   1. ~~~bash
      screen -S env_name
      sudo apt install screen
      screen -ls
      screen -r env_name
      screen -X -S env_name_delete
      ~~~
   2. Appimage文件执行

      1. ~~~bash
         sudo apt install libfuse2
         chmod -X file
         ./file_name.Appimage
         ~~~
   3. deb文件执行

      1. ~~~bash
         sudo dpkg -i file
         ~~~
4. 拼音的安装

   1. 进入linux系统桌面后，点击左侧导航栏show apps
   2. 选择setting，在左侧寻找system，点击region and language
   3. 选择manage install language，在弹出的设置中选择install/remove language，找到Chinese(simplified)
   4. 点击Apply system-wide
   5. 随后返回setting，找到keyboard，在终端中输入如下命令

   ~~~bash
   sudo apt install ibus-pinyin
   ~~~

   点击keyboard中的+号，搜索Chinese(intelligent)，即可完成中文输入法安装
5.
