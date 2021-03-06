import torch
import matchzoo as mz

# rank任务
# RankCrossEntropyLoss：分类问题，ylog(y)
ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=4))
# evaluation metrics（标准）
# NormalizedDiscountedCumulativeGain：排名越靠前的结果越能影响最后的结果
# MeanAveragePrecision：pr曲线下的面积
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.MeanAveragePrecision()
]

train_pack = mz.datasets.wiki_qa.load_data('train', task=ranking_task)
valid_pack = mz.datasets.wiki_qa.load_data('dev', task=ranking_task)

preprocessor = mz.models.ArcI.get_default_preprocessor()
train_processed = preprocessor.fit_transform(train_pack)
valid_processed = preprocessor.transform(valid_pack)

trainset = mz.dataloader.Dataset(
    data_pack=train_processed,
    mode='pair',
    num_dup=1,
    num_neg=4,
    batch_size=32
)

validset = mz.dataloader.Dataset(
    data_pack=valid_processed,
    mode='point',
    batch_size=32
)

padding_callback = mz.models.ArcI.get_default_padding_callback()

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    stage='train',
    callback=padding_callback
)
validloader = mz.dataloader.DataLoader(
    dataset=validset,
    stage='dev',
    callback=padding_callback
)

model = mz.models.ArcI()
model.params['task'] = ranking_task
model.params['embedding_output_dim'] = 100
model.params['embedding_input_dim'] = preprocessor.context['embedding_input_dim']
model.guess_and_fill_missing_params()
model.build()

optimizer = torch.optim.Adam(model.parameters())
trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    epochs=10
)
trainer.run()
