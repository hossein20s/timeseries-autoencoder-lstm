from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext


def hbase_connect():
    # conf = SparkConf()
    # conf.setMaster("c886adf6a242")
    # conf.set("spark.ui.port","7077")
    # conf.set('spark.hbase.host', 'hbase')

    sc = SparkContext()

    print(sc.getConf().getAll())
    conf = sc.getConf()
    sqlc = SQLContext(sc)


    data_source_format = 'org.apache.hadoop.hbase.spark'

    df = sc.parallelize([('a', '1.0'), ('b', '2.0')]).toDF(schema=['col0', 'col1'])

    # ''.join(string.split()) in order to write a multi-line JSON string here.
    catalog = ''.join("""{
        "table":{"namespace":"default", "name":"testtable"},
        "rowkey":"key",
        "columns":{
            "col0":{"cf":"rowkey", "col":"key", "type":"string"},
            "col1":{"cf":"cf", "col":"col1", "type":"string"}
        }
    }""".split())

    

    # Writing
    #df.write.options(catalog=catalog).format(data_source_format).save()
    # alternatively: .option('catalog', catalog)

    # Reading
    #df = sqlc.read.options(catalog=catalog).format(data_source_format).load()

if __name__ == '__main__':
    hbase_connect()
