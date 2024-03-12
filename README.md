# 基本操作

1. 推荐使用[操作指南](https://github.com/iiisthu/ailab?tab=readme-ov-file#使用-vs-code-连接k8s远程调试)中的“使用 VS Code 连接K8S远程调试”方法先在vscode中连接集群。
2. 在VS Code命令行（terminal）中，clone课程仓库：
    ```
    export http_proxy=http://Clash:QOAF8Rmd@10.1.0.213:7890 && export https_proxy=http://Clash:QOAF8Rmd@10.1.0.213:7890 && export all_proxy=socks5://Clash:QOAF8Rmd@10.1.0.213:7893
    git clone git@github.com:xuw/llm_course_public.git
    ```
4. (更新)查看最新的课程内容信息：
    ```
    cd llm_course_public/ && git pull --all
    ```
5. 在VS Code中运行labs的Jupyter Notebook
    - 确定在Server端Jupyter插件已正确安装，已经启用（enable）
    - 在GUI中设置kernel（environment）为conda即可
