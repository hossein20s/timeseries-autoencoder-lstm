from pyspark.shell import spark
from delta.tables import *
# add /opt/packages/delta/python/ this to content root
from pyspark.sql.functions import *


def write():
    data = spark.range(0, 5)
    data.write.format("delta").save("/tmp/delta-table")


def read():
    df = spark.read.format("delta").load("/tmp/delta-table")
    df.show()


def update():
    data = spark.range(5, 10)
    data.write.format("delta").mode("overwrite").save("/tmp/delta-table")

def conditional_update():
    deltaTable = DeltaTable.forPath(spark, "/tmp/delta-table")

    # Update every even value by adding 100 to it
    deltaTable.update(
        condition=expr("id % 2 == 0"),
        set={"id": expr("id + 100")})

    # Delete every even value
    deltaTable.delete(condition=expr("id % 2 == 0"))

    # Upsert (merge) new data
    newData = spark.range(0, 20)

    deltaTable.alias("oldData") \
        .merge(
        newData.alias("newData"),
        "oldData.id = newData.id") \
        .whenMatchedUpdate(set={"id": col("newData.id")}) \
        .whenNotMatchedInsert(values={"id": col("newData.id")}) \
        .execute()

    deltaTable.toDF().show()

if __name__ == '__main__':
    update()
