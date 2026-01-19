# 知识库维护指南

本文档定义了 PNCG_IPC 项目的知识库组织结构和维护规范。

## 目录结构

```
PNCG_IPC/
├── CLAUDE.md                    # Claude Code 指令（入口点）
├── README.md                    # 项目简介（面向外部用户）
├── docs/                        # 知识库主目录
│   ├── INDEX.md                 # 文档索引
│   ├── KNOWLEDGE_BASE_GUIDE.md  # 本文档
│   ├── algorithm/               # 算法实现文档
│   ├── design/                  # 设计与规划文档
│   ├── reference/               # 参考实现分析
│   └── papers/                  # 论文PDF和tex源文件
└── experiment_reports/          # 实验报告（持续更新）
    └── *.md                     # 按日期/主题命名的报告
```

## 文档分类

### 1. CLAUDE.md（入口点）
- **用途**: Claude Code 的主要指令文件
- **内容**: 项目概览、快速命令、关键参数
- **原则**: 保持精简（<150行），只放最常用信息
- **更新频率**: 架构变更时更新

### 2. docs/algorithm/（算法实现）
- **用途**: 核心算法的实现细节
- **命名规范**: `{模块名}_IMPLEMENTATION.md`
- **内容**: 数据结构、算法流程、接口说明
- **更新频率**: 代码重构或算法修改时更新

### 3. docs/design/（设计文档）
- **用途**: 架构设计、集成方案、优化计划
- **命名规范**: `{模块名}_DESIGN.md` 或 `{模块名}_TODO.md`
- **更新频率**: 规划新功能或架构变更时更新

### 4. docs/reference/（参考实现）
- **用途**: 外部参考实现的分析文档
- **命名规范**: `{项目名}_IMPLEMENTATION_DETAILS.md`
- **更新频率**: 研究新方法时更新

### 5. docs/papers/（论文资料）
- **用途**: 相关论文PDF和LaTeX源文件
- **内容**: 论文、补充材料、tex源文件
- **更新频率**: 较少更新

### 6. experiment_reports/（实验报告）
- **用途**: 记录实验过程、结果、结论
- **命名规范**: `{YYYY-MM-DD}_{主题}.md` 或 `{模块}_{实验名}.md`
- **更新频率**: 每次重要实验后立即创建

## 文档模板

### 实验报告模板
```markdown
# {实验标题}

**日期**: YYYY-MM-DD
**作者**: {作者}
**状态**: 进行中 / 已完成

## 目的
简要说明实验目标。

## 方法
- 修改内容
- 测试配置

## 结果
| 指标 | 修改前 | 修改后 |
|------|--------|--------|
| ...  | ...    | ...    |

## 结论
关键发现和后续建议。

## 相关文件
- `path/to/modified/file.py`
```

### 实现文档模板
```markdown
# {模块名} 实现文档

## 概述
模块功能简介。

## 核心数据结构
关键类和字段说明。

## 主要算法
算法流程和复杂度。

## 接口说明
公共 API 文档。

## 依赖关系
与其他模块的关系。
```

## 维护规则

### 何时更新文档

| 事件 | 更新内容 |
|------|----------|
| 完成重要实验 | 创建 `experiment_reports/{日期}_{主题}.md` |
| 修改核心算法 | 更新 `docs/algorithm/{模块}_IMPLEMENTATION.md` |
| 重构代码架构 | 更新 `CLAUDE.md` 和相关设计文档 |
| 添加新模块 | 创建 `docs/algorithm/{模块}_IMPLEMENTATION.md` |
| 修复重要 bug | 在实验报告中记录 |
| 规划新功能 | 创建 `docs/design/{功能}_DESIGN.md` |

### 文档质量要求

1. **及时性**: 代码修改后立即更新文档
2. **准确性**: 文档必须与代码保持一致
3. **简洁性**: 避免冗余，删除过时内容
4. **可追溯**: 记录修改日期和原因

### 定期维护

- **周维护**: 检查 experiment_reports 是否有未归档内容
- **月维护**: 审查 docs/ 下文档与代码一致性
- **季度维护**: 清理过时文档，更新 INDEX.md

## Claude Code 集成

在 CLAUDE.md 中引用文档时使用相对路径：

```markdown
### Reference Documents
- `docs/algorithm/MAS_PRECONDITIONER_IMPLEMENTATION.md` - MAS实现细节
- `docs/papers/MAS_PNCG_clean.tex` - 论文源文件
- `experiment_reports/` - 实验报告目录
```

## 快速命令

```bash
# 创建新实验报告
touch experiment_reports/$(date +%Y-%m-%d)_experiment_name.md

# 查看所有文档
find docs -name "*.md" -type f

# 检查文档更新时间
ls -lt docs/**/*.md | head -10

# 查看文档索引
cat docs/INDEX.md
```
