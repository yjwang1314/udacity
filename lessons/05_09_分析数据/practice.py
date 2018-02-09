#!/usr/bin/env python
"""
In an earlier exercise we looked at the cities dataset and asked which region in India contains 
the most cities. In this exercise, we'd like you to answer a related question regarding regions in 
India. What is the average city population for a region in India? Calculate your answer by first 
finding the average population of cities in each region and then by calculating the average of the 
regional averages.

Hint: If you want to accumulate using values from all input documents to a group stage, you may use 
a constant as the value of the "_id" field. For example, 
    { "$group" : {"_id" : "India Regional City Population Average",
      ... }

Please modify only the 'make_pipeline' function so that it creates and returns an aggregation 
pipeline that can be passed to the MongoDB aggregate function. As in our examples in this lesson, 
the aggregation pipeline should be a list of one or more dictionary objects. 
Please review the lesson examples if you are unsure of the syntax.

Your code will be run against a MongoDB instance that we have provided. If you want to run this code 
locally on your machine, you have to install MongoDB, download and insert the dataset.
For instructions related to MongoDB setup and datasets please see Course Materials.

Please note that the dataset you are using here is a smaller version of the twitter dataset used 
in examples in this lesson. If you attempt some of the same queries that we looked at in the lesson 
examples, your results will be different.

在上一道练习中，我们查看了城市数据集，并询问印度的哪个地区包含的城市最多。在这道练习中，我们想请你回答另一个关于印度地区的相关问题。印度各个地区的平均人口数量是多少？你需要首先计算每个地区城市的平均人口数量，然后计算地区的平均人口数量。

提示：如果你想使用所有输入文档中的值汇集到一个群组阶段中，可以使用常量作为“_id”字段的值。例如：
{ "$group" : {"_id" : "India Regional City Population Average", ... }

只需修改“make_pipeline”函数，使其创建并返回一个聚合管道，该管道可以传递到 MongoDB 聚合函数中。和这节课中的示例一样，聚合管道应该是一个包含一个或多个字典对象的列表。如果不熟悉语法，请参阅这节课中的示例。

你的代码将根据我们提供的 MongoDB 实例运行。如果你想在本地机器上运行代码，你需要安装 MongoDB 并下载和插入数据集。要了解 MongoDB 设置和数据集方面的说明，请参阅课程资料。

请注意，你在此处使用的数据集是这节课的示例中使用的推特数据集的简略版本。如果你尝试运行我们在课程示例中运行过的同一查询，结果可能不同。

{
    "_id" : ObjectId("52fe1d364b5ab856eea75ebc"),
    "elevation" : 1855,
    "name" : "Kud",
    "country" : "India",
    "lon" : 75.28,
    "lat" : 33.08,
    "isPartOf" : [
        "Jammu and Kashmir",
        "Udhampur district"
    ],
    "timeZone" : [
        "Indian Standard Time"
    ],
    "population" : 1140
}
"""

def get_db(db_name):
    from pymongo import MongoClient
    client = MongoClient('localhost:27017')
    db = client[db_name]
    return db

def make_pipeline():
    # complete the aggregation pipeline
    pipeline = [{'$match':{'county':'India'}},
                {'$unwind':'$isPartOf'},
                {"$group":{"_id":"$isPartOf",'avg_pop':{'$avg':'$population'}}},
                {'$group':{'_id':'India Regional City Population Average', 'avg':{'$avg':'$avg_pop'}}}]
    return pipeline

def aggregate(db, pipeline):
    return [doc for doc in db.cities.aggregate(pipeline)]


if __name__ == '__main__':
    db = get_db('examples')
    pipeline = make_pipeline()
    result = aggregate(db, pipeline)
    assert len(result) == 1
    # Your result should be close to the value after the minus sign.
    assert abs(result[0]["avg"] - 201128.0241546919) < 10 ** -8
    import pprint
    pprint.pprint(result)
