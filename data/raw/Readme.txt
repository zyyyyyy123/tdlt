请使用pandas工具包读取pkl文件，其中Metrics/loss列为对应的loss值， lr列表示对应的学习率。

注：其中的``8-1-1'' LRS表示按照总的training steps，在80%处，90%处，学习率都decay为原来上一阶段的1/\sqrt{10}，也即是假设最大学习率为\eta，那么总共的three-stage学习率就为：\eta, \eta/\sqrt{10}, \eta/10。

数据清洗记录：

- gpt_loss+lrs.pkl 和 gpt_loss+lrs.csv 已清洗为每条 run 都包含连续 step 0..33907，共 33908 个点。
- 对原始数据中缺失的 step 使用 forward fill：新增行的 Metrics/loss 和 lr 都复制上一 step 的值。
- 已填补的缺失点包括：
  - scheduler:wsd_rope 的 step=20815，复制 step=20814。
  - scheduler:cosine_rope 的 step=22493，复制 step=22492。
- pkl_to_csv.py 会读取 gpt_loss+lrs.pkl，检查 step 范围、重复 step 和关键列 NaN，补齐缺失 step 后回写 pkl 并重新生成 csv。
