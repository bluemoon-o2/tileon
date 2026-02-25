---
title: Contributing
---

# 贡献指南

感谢您对 Tileon 项目的关注！本文档提供贡献项目的指南。

## 行为准则

参与本项目即表示您同意遵守我们的行为准则。

## 如何贡献

### 报告问题

如果您发现 bug 或有功能请求，请在 GitHub 上开 issue。报告问题时，请包含：

- 问题的清晰描述
- 复现问题的步骤
- 预期行为
- 实际行为
- 您的环境（操作系统、Python 版本、GPU 等）

### 提交 Pull Request

1. Fork 仓库
2. 为您的功能或 bug 修复创建新分支
3. 进行更改
4. 确保代码通过所有测试
5. 提交 Pull Request

#### Pull Request 指南

- 遵循现有代码风格
- 编写清晰的提交信息
- 为新功能或 bug 修复添加测试
- 保持 Pull Request 专注于单一更改

## 开发环境设置

### 前置条件

- Python 3.10+
- CMake 3.20+
- Ninja
- C++ 编译器

### 从源码构建

```bash
# 克隆仓库
git clone https://github.com/bluemoon-o2/tileon.git
cd tileon

# 以开发模式安装
pip install -e .
```

### 运行测试

```bash
# 运行 Python 测试
pytest python/test/
```

## 代码风格

我们使用 pre-commit 钩子来强制执行代码风格。请在提交更改之前安装并运行 pre-commit：

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## 获取帮助

如果您有疑问或需要帮助，请在 GitHub 上开 issue 或讨论。

## 许可证

为 Tileon 贡献代码即表示您同意您的贡献将基于 MIT 许可证授权。
