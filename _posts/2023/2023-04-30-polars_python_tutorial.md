---
title: Hands on tutorial with Polars DataFrames in Python
date: 2023-04-30
tags:
- python
excerpt: "Tutorial to work on large datasets in memory Polars DataFrames in Python"
---

## Introduction

Handling large datasets locally on laptops is not easy. As a workaround, data scientists either work on a sample data set or spin up spark clusters. Spark clusters don't come cheap. I recently migrated a data science project from spark to python in-memory computation using polars and saved few hunderd dollars weekly cloud cost. Yay!

Polars is just like pandas (used to read, manipulate, write datasets), but executes much faster in memory. I've been using pandas for several years now, so learning to use polars wasn't that difficult. I found the polars api is quite intuitive and as a data scientist you'll find it quite useful in your day to day data analysis, machine learning projects while working on big datasets.

Polars to pandas transition is super easy, just doing `.to_pandas()` on a polars dataframe brings us back to using pandas. 

Some key features about polars library are:

* It is written in [Rust](https://stackoverflow.blog/2020/01/20/what-is-rust-and-why-is-it-so-popular/) based on [Apache Columnar Format](https://arrow.apache.org/docs/format/Columnar.html).
* It allows lazy execution of commands, thus allowing it to handle dataset larger than RAM on your laptop. 
* Its API is very similar to pandas and pyspark, thus the learning curve is not at all steep.
* If you are more comfortable with SQL, you can easily convert a polar dataframe into a SQL Table locally and run SQL commands normally.
* It is heavily optimised for doing parallel computations. 


In this tutorial, I'll show how to do some commonly used dataframes calculations/transformations using polars in Python.

### Table of Contents


1. [How to get count of NA values in the polar dataframe?](####-Q.-How-to-get-count-of-NA-values-in-the-polar-dataframe?)
2. [How to run groupby functions?](####-Q.-How-to-run-groupby-functions?)
3. [How to set the data types correctly?](####-Q.-How-to-set-the-data-types-correctly-?)
4. [How to find number of groups in a column?](####-Q.-How-to-find-number-of-groups-in-a-column-?)
5. [How to do one hot encoding in polars dataframe?](####-Q.-How-to-do-One-Hot-Encoding-in-polars-dataframe? )
6. [How to create categories/bins using a numeric column?](####-Q.-How-to-create-categories/bins-using-a-numeric-column?)
7. [How to create a pivot table ?](####-Q.-How-to-create-a-pivot-table-?)
8. [How to create a column where each row contains multiple values?](####-Q.-How-to-create-a-column-where-each-row-contains-multiple-values?)
9. [How to convert dataframe columns into a python dict?](####-Q.-How-to-convert-dataframe-columns-into-a-python-dict-?)
10. [How to convert a column containing list into separate rows?](####-Q.-How-to-convert-a-column-containing-list-into-separate-rows?)
11. [How to perform string related functions using polars?](####-Q.-How-to-perform-string-related-functions-using-polars?)
12. [How to perform datetime functions using polars?](####-Q.-How-to-perform-datetime-functions-using-polars?)
13. [How to read parquet data from S3 with polars?](####-Q-How-to-read-parquet-data-from-S3-with-polars?)



Lets organise the imports we require in this tutorial. Also, lets create a sample data set which we are going to use for this tutorial.


```python
import polars as pl
import boto3
import datetime
```


```python
# set the configs
pl.Config.set_fmt_str_lengths(100)
pl.Config.set_tbl_width_chars(100)
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_hide_dataframe_shape(True)  
```




    polars.config.Config




```python
# create a polars dataframe
df = pl.DataFrame({
    "group": ["one", "one", "one", "two", "two", "three", "three"],
    "grade": ["1", "98", "2", "3", "99", "23", "11"],
    "score": [45,49,76,83,69,90,80],
    "date1": pl.date_range(datetime.date(2023, 1, 1), datetime.date(2023, 1, 7)),
    "date2": pl.date_range(datetime.date(2023, 2, 1), datetime.date(2023, 2, 7)),
    "rating": [4.5, 3.5, 4, 1.0, 1.2, 9.4, 9.1],
    "zone": [None, float('nan'), 1, 2, 4, 5, 1],
    "class": ['x1', float('nan'), None, 'c1', 'c2', 'x2', 'j1'],
})

```


```python
df.head()
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (5, 8)</small><table class="dataframe"><thead><tr><th>group</th><th>grade</th><th>score</th><th>date1</th><th>date2</th><th>rating</th><th>zone</th><th>class</th></tr><tr><td>str</td><td>str</td><td>i64</td><td>date</td><td>date</td><td>f64</td><td>f64</td><td>str</td></tr></thead><tbody><tr><td>&quot;one&quot;</td><td>&quot;1&quot;</td><td>45</td><td>2023-01-01</td><td>2023-02-01</td><td>4.5</td><td>null</td><td>&quot;x1&quot;</td></tr><tr><td>&quot;one&quot;</td><td>&quot;98&quot;</td><td>49</td><td>2023-01-02</td><td>2023-02-02</td><td>3.5</td><td>NaN</td><td>null</td></tr><tr><td>&quot;one&quot;</td><td>&quot;2&quot;</td><td>76</td><td>2023-01-03</td><td>2023-02-03</td><td>4.0</td><td>1.0</td><td>null</td></tr><tr><td>&quot;two&quot;</td><td>&quot;3&quot;</td><td>83</td><td>2023-01-04</td><td>2023-02-04</td><td>1.0</td><td>2.0</td><td>&quot;c1&quot;</td></tr><tr><td>&quot;two&quot;</td><td>&quot;99&quot;</td><td>69</td><td>2023-01-05</td><td>2023-02-05</td><td>1.2</td><td>4.0</td><td>&quot;c2&quot;</td></tr></tbody></table></div>




```python
df.glimpse()
```

    Rows: 7
    Columns: 8
    $ group   <str> one, one, one, two, two, three, three
    $ grade   <str> 1, 98, 2, 3, 99, 23, 11
    $ score   <i64> 45, 49, 76, 83, 69, 90, 80
    $ date1  <date> 2023-01-01, 2023-01-02, 2023-01-03, 2023-01-04, 2023-01-05, 2023-01-06, 2023-01-07
    $ date2  <date> 2023-02-01, 2023-02-02, 2023-02-03, 2023-02-04, 2023-02-05, 2023-02-06, 2023-02-07
    $ rating  <f64> 4.5, 3.5, 4.0, 1.0, 1.2, 9.4, 9.1
    $ zone    <f64> None, nan, 1.0, 2.0, 4.0, 5.0, 1.0
    $ class   <str> x1, None, None, c1, c2, x2, j1
    



```python
df.describe()
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (9, 9)</small><table class="dataframe"><thead><tr><th>describe</th><th>group</th><th>grade</th><th>score</th><th>date1</th><th>date2</th><th>rating</th><th>zone</th><th>class</th></tr><tr><td>str</td><td>str</td><td>str</td><td>f64</td><td>str</td><td>str</td><td>f64</td><td>f64</td><td>str</td></tr></thead><tbody><tr><td>&quot;count&quot;</td><td>&quot;7&quot;</td><td>&quot;7&quot;</td><td>7.0</td><td>&quot;7&quot;</td><td>&quot;7&quot;</td><td>7.0</td><td>7.0</td><td>&quot;7&quot;</td></tr><tr><td>&quot;null_count&quot;</td><td>&quot;0&quot;</td><td>&quot;0&quot;</td><td>0.0</td><td>&quot;0&quot;</td><td>&quot;0&quot;</td><td>0.0</td><td>1.0</td><td>&quot;2&quot;</td></tr><tr><td>&quot;mean&quot;</td><td>null</td><td>null</td><td>70.285714</td><td>null</td><td>null</td><td>4.671429</td><td>NaN</td><td>null</td></tr><tr><td>&quot;std&quot;</td><td>null</td><td>null</td><td>17.182494</td><td>null</td><td>null</td><td>3.39986</td><td>NaN</td><td>null</td></tr><tr><td>&quot;min&quot;</td><td>&quot;one&quot;</td><td>&quot;1&quot;</td><td>45.0</td><td>&quot;2023-01-01&quot;</td><td>&quot;2023-02-01&quot;</td><td>1.0</td><td>1.0</td><td>&quot;c1&quot;</td></tr><tr><td>&quot;max&quot;</td><td>&quot;two&quot;</td><td>&quot;99&quot;</td><td>90.0</td><td>&quot;2023-01-07&quot;</td><td>&quot;2023-02-07&quot;</td><td>9.4</td><td>5.0</td><td>&quot;x2&quot;</td></tr><tr><td>&quot;median&quot;</td><td>null</td><td>null</td><td>76.0</td><td>null</td><td>null</td><td>4.0</td><td>3.0</td><td>null</td></tr><tr><td>&quot;25%&quot;</td><td>null</td><td>null</td><td>49.0</td><td>null</td><td>null</td><td>1.2</td><td>1.0</td><td>null</td></tr><tr><td>&quot;75%&quot;</td><td>null</td><td>null</td><td>83.0</td><td>null</td><td>null</td><td>9.1</td><td>5.0</td><td>null</td></tr></tbody></table></div>





#### Q. How to get count of NA values in the polar dataframe? 


```python
df.select(pl.all().is_null().sum())
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (1, 8)</small><table class="dataframe"><thead><tr><th>group</th><th>grade</th><th>score</th><th>date1</th><th>date2</th><th>rating</th><th>zone</th><th>class</th></tr><tr><td>u32</td><td>u32</td><td>u32</td><td>u32</td><td>u32</td><td>u32</td><td>u32</td><td>u32</td></tr></thead><tbody><tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>2</td></tr></tbody></table></div>





#### Q. How to run groupby functions? 

Probably, `groupby` is one the most commonly used data manipulation functions. Polars offers `.groupby` and `.over` functions to calculate aggregated metrics over a group.


```python
# method 1: find the first row in each group sorted by some value
df.groupby("group").agg(pl.all().sort_by("score",descending=True).first())
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (3, 8)</small><table class="dataframe"><thead><tr><th>group</th><th>grade</th><th>score</th><th>date1</th><th>date2</th><th>rating</th><th>zone</th><th>class</th></tr><tr><td>str</td><td>str</td><td>i64</td><td>date</td><td>date</td><td>f64</td><td>f64</td><td>str</td></tr></thead><tbody><tr><td>&quot;one&quot;</td><td>&quot;2&quot;</td><td>76</td><td>2023-01-03</td><td>2023-02-03</td><td>4.0</td><td>1.0</td><td>null</td></tr><tr><td>&quot;two&quot;</td><td>&quot;3&quot;</td><td>83</td><td>2023-01-04</td><td>2023-02-04</td><td>1.0</td><td>2.0</td><td>&quot;c1&quot;</td></tr><tr><td>&quot;three&quot;</td><td>&quot;23&quot;</td><td>90</td><td>2023-01-06</td><td>2023-02-06</td><td>9.4</td><td>5.0</td><td>&quot;x2&quot;</td></tr></tbody></table></div>




```python
# method 2: find the first row in each group sorted by some value
df.filter(pl.col("score") == pl.col("score").max().over(["group"]))
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (3, 8)</small><table class="dataframe"><thead><tr><th>group</th><th>grade</th><th>score</th><th>date1</th><th>date2</th><th>rating</th><th>zone</th><th>class</th></tr><tr><td>str</td><td>str</td><td>i64</td><td>date</td><td>date</td><td>f64</td><td>f64</td><td>str</td></tr></thead><tbody><tr><td>&quot;one&quot;</td><td>&quot;2&quot;</td><td>76</td><td>2023-01-03</td><td>2023-02-03</td><td>4.0</td><td>1.0</td><td>null</td></tr><tr><td>&quot;two&quot;</td><td>&quot;3&quot;</td><td>83</td><td>2023-01-04</td><td>2023-02-04</td><td>1.0</td><td>2.0</td><td>&quot;c1&quot;</td></tr><tr><td>&quot;three&quot;</td><td>&quot;23&quot;</td><td>90</td><td>2023-01-06</td><td>2023-02-06</td><td>9.4</td><td>5.0</td><td>&quot;x2&quot;</td></tr></tbody></table></div>




```python
# calculate mean score over group and add it as a column in the dataframe
df = df.with_columns(mean_over_grp=pl.col("score").sum().over("group"))
df.head(5)
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (5, 9)</small><table class="dataframe"><thead><tr><th>group</th><th>grade</th><th>score</th><th>date1</th><th>date2</th><th>rating</th><th>zone</th><th>class</th><th>mean_over_grp</th></tr><tr><td>str</td><td>str</td><td>i64</td><td>date</td><td>date</td><td>f64</td><td>f64</td><td>str</td><td>i64</td></tr></thead><tbody><tr><td>&quot;one&quot;</td><td>&quot;1&quot;</td><td>45</td><td>2023-01-01</td><td>2023-02-01</td><td>4.5</td><td>null</td><td>&quot;x1&quot;</td><td>170</td></tr><tr><td>&quot;one&quot;</td><td>&quot;98&quot;</td><td>49</td><td>2023-01-02</td><td>2023-02-02</td><td>3.5</td><td>NaN</td><td>null</td><td>170</td></tr><tr><td>&quot;one&quot;</td><td>&quot;2&quot;</td><td>76</td><td>2023-01-03</td><td>2023-02-03</td><td>4.0</td><td>1.0</td><td>null</td><td>170</td></tr><tr><td>&quot;two&quot;</td><td>&quot;3&quot;</td><td>83</td><td>2023-01-04</td><td>2023-02-04</td><td>1.0</td><td>2.0</td><td>&quot;c1&quot;</td><td>152</td></tr><tr><td>&quot;two&quot;</td><td>&quot;99&quot;</td><td>69</td><td>2023-01-05</td><td>2023-02-05</td><td>1.2</td><td>4.0</td><td>&quot;c2&quot;</td><td>152</td></tr></tbody></table></div>




```python
# convert a column into a list per group
df.groupby("group", maintain_order=True).agg(score_list=pl.col("score").apply(list))
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (3, 2)</small><table class="dataframe"><thead><tr><th>group</th><th>score_list</th></tr><tr><td>str</td><td>list[i64]</td></tr></thead><tbody><tr><td>&quot;one&quot;</td><td>[45, 49, 76]</td></tr><tr><td>&quot;two&quot;</td><td>[83, 69]</td></tr><tr><td>&quot;three&quot;</td><td>[90, 80]</td></tr></tbody></table></div>




```python
# aggregate calculations for a group using multiple columns
df.groupby('group').agg(
    mean_score = pl.col('score').mean(), 
    n_rows = pl.count(), 
    cls_c1_count = (pl.col('class') == 'c1').sum(),
    min_date = pl.col('date1').min()
)
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (3, 5)</small><table class="dataframe"><thead><tr><th>group</th><th>mean_score</th><th>n_rows</th><th>cls_c1_count</th><th>min_date</th></tr><tr><td>str</td><td>f64</td><td>u32</td><td>u32</td><td>date</td></tr></thead><tbody><tr><td>&quot;two&quot;</td><td>76.0</td><td>2</td><td>1</td><td>2023-01-04</td></tr><tr><td>&quot;one&quot;</td><td>56.666667</td><td>3</td><td>0</td><td>2023-01-01</td></tr><tr><td>&quot;three&quot;</td><td>85.0</td><td>2</td><td>0</td><td>2023-01-06</td></tr></tbody></table></div>




```python
# method 1: calculate counts per group
df.groupby('group').count()
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (3, 2)</small><table class="dataframe"><thead><tr><th>group</th><th>count</th></tr><tr><td>str</td><td>u32</td></tr></thead><tbody><tr><td>&quot;two&quot;</td><td>2</td></tr><tr><td>&quot;one&quot;</td><td>3</td></tr><tr><td>&quot;three&quot;</td><td>2</td></tr></tbody></table></div>




```python
# method 2: calculate counts per group
df['group'].value_counts()
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (3, 2)</small><table class="dataframe"><thead><tr><th>group</th><th>counts</th></tr><tr><td>str</td><td>u32</td></tr></thead><tbody><tr><td>&quot;two&quot;</td><td>2</td></tr><tr><td>&quot;one&quot;</td><td>3</td></tr><tr><td>&quot;three&quot;</td><td>2</td></tr></tbody></table></div>





#### Q. How to set the data types correctly ? 

We find the `grade` column has integers but encoded as strings in the dataframe. Lets fix the data type.


```python
## convert string to integer
df = df.with_columns(pl.col('grade').cast(pl.Int16))

# similarly to convert an integer to string you can do:
df = df.with_columns(pl.col('score').cast(pl.Utf8))

df['grade'].head(3)
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (3,)</small><table class="dataframe"><thead><tr><th>grade</th></tr><tr><td>i16</td></tr></thead><tbody><tr><td>1</td></tr><tr><td>98</td></tr><tr><td>2</td></tr></tbody></table></div>





#### Q. How to find number of groups in a column ? 

The `n_groups` function in pandas returns the number of unique groups in a column. We can achieve that using `n_unique` in polars.


```python
# find unique groups in group col
df.select(pl.col("group").n_unique()).item()
```




    3




```python
# find unique groups in group and class togther
df.select(pl.struct(["group", "class"]).n_unique()).item()
```




    6





#### Q. How to do One Hot Encoding in polars dataframe?

Converting a categorical column into several column with 1 and 0 is a commonly used feature engineering technique.


```python
df=df.to_dummies('group')
df.head()
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (7, 11)</small><table class="dataframe"><thead><tr><th>group_one</th><th>group_three</th><th>group_two</th><th>grade</th><th>score</th><th>date1</th><th>date2</th><th>rating</th><th>zone</th><th>class</th><th>mean_over_grp</th></tr><tr><td>u8</td><td>u8</td><td>u8</td><td>i16</td><td>str</td><td>date</td><td>date</td><td>f64</td><td>f64</td><td>str</td><td>i64</td></tr></thead><tbody><tr><td>1</td><td>0</td><td>0</td><td>1</td><td>&quot;45&quot;</td><td>2023-01-01</td><td>2023-02-01</td><td>4.5</td><td>null</td><td>&quot;x1&quot;</td><td>170</td></tr><tr><td>1</td><td>0</td><td>0</td><td>98</td><td>&quot;49&quot;</td><td>2023-01-02</td><td>2023-02-02</td><td>3.5</td><td>NaN</td><td>null</td><td>170</td></tr><tr><td>1</td><td>0</td><td>0</td><td>2</td><td>&quot;76&quot;</td><td>2023-01-03</td><td>2023-02-03</td><td>4.0</td><td>1.0</td><td>null</td><td>170</td></tr><tr><td>0</td><td>0</td><td>1</td><td>3</td><td>&quot;83&quot;</td><td>2023-01-04</td><td>2023-02-04</td><td>1.0</td><td>2.0</td><td>&quot;c1&quot;</td><td>152</td></tr><tr><td>0</td><td>0</td><td>1</td><td>99</td><td>&quot;69&quot;</td><td>2023-01-05</td><td>2023-02-05</td><td>1.2</td><td>4.0</td><td>&quot;c2&quot;</td><td>152</td></tr><tr><td>0</td><td>1</td><td>0</td><td>23</td><td>&quot;90&quot;</td><td>2023-01-06</td><td>2023-02-06</td><td>9.4</td><td>5.0</td><td>&quot;x2&quot;</td><td>170</td></tr><tr><td>0</td><td>1</td><td>0</td><td>11</td><td>&quot;80&quot;</td><td>2023-01-07</td><td>2023-02-07</td><td>9.1</td><td>1.0</td><td>&quot;j1&quot;</td><td>170</td></tr></tbody></table></div>





#### Q. How to create categories/bins using a numeric column?

Here, we'd like to create buckets of numbers based on percentiles inside a group.


```python
df=df.with_columns(
    score_bin=pl.col("score")
    .apply(
        lambda s: s.qcut(
            [0.2, 0.4, 0.6, 0.8], labels=["1", "2", "3", "4", "5"], maintain_order=True
        )["category"]
    )
    .over(["group"])
)
df
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (7, 10)</small><table class="dataframe"><thead><tr><th>group</th><th>grade</th><th>score</th><th>date1</th><th>date2</th><th>rating</th><th>zone</th><th>class</th><th>mean_over_grp</th><th>score_bin</th></tr><tr><td>str</td><td>i16</td><td>str</td><td>date</td><td>date</td><td>f64</td><td>f64</td><td>str</td><td>i64</td><td>cat</td></tr></thead><tbody><tr><td>&quot;one&quot;</td><td>1</td><td>&quot;45&quot;</td><td>2023-01-01</td><td>2023-02-01</td><td>4.5</td><td>null</td><td>&quot;x1&quot;</td><td>170</td><td>&quot;1&quot;</td></tr><tr><td>&quot;one&quot;</td><td>98</td><td>&quot;49&quot;</td><td>2023-01-02</td><td>2023-02-02</td><td>3.5</td><td>NaN</td><td>null</td><td>170</td><td>&quot;3&quot;</td></tr><tr><td>&quot;one&quot;</td><td>2</td><td>&quot;76&quot;</td><td>2023-01-03</td><td>2023-02-03</td><td>4.0</td><td>1.0</td><td>null</td><td>170</td><td>&quot;5&quot;</td></tr><tr><td>&quot;two&quot;</td><td>3</td><td>&quot;83&quot;</td><td>2023-01-04</td><td>2023-02-04</td><td>1.0</td><td>2.0</td><td>&quot;c1&quot;</td><td>152</td><td>&quot;5&quot;</td></tr><tr><td>&quot;two&quot;</td><td>99</td><td>&quot;69&quot;</td><td>2023-01-05</td><td>2023-02-05</td><td>1.2</td><td>4.0</td><td>&quot;c2&quot;</td><td>152</td><td>&quot;1&quot;</td></tr><tr><td>&quot;three&quot;</td><td>23</td><td>&quot;90&quot;</td><td>2023-01-06</td><td>2023-02-06</td><td>9.4</td><td>5.0</td><td>&quot;x2&quot;</td><td>170</td><td>&quot;5&quot;</td></tr><tr><td>&quot;three&quot;</td><td>11</td><td>&quot;80&quot;</td><td>2023-01-07</td><td>2023-02-07</td><td>9.1</td><td>1.0</td><td>&quot;j1&quot;</td><td>170</td><td>&quot;1&quot;</td></tr></tbody></table></div>





#### Q. How to create a pivot table ? 


```python
df.pivot(values="rating", index="score_bin", columns="group", aggregate_function="min")
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (3, 4)</small><table class="dataframe"><thead><tr><th>score_bin</th><th>one</th><th>two</th><th>three</th></tr><tr><td>cat</td><td>f64</td><td>f64</td><td>f64</td></tr></thead><tbody><tr><td>&quot;1&quot;</td><td>4.5</td><td>1.2</td><td>9.1</td></tr><tr><td>&quot;3&quot;</td><td>3.5</td><td>null</td><td>null</td></tr><tr><td>&quot;5&quot;</td><td>4.0</td><td>1.0</td><td>9.4</td></tr></tbody></table></div>





#### Q. How to create a column where each row contains multiple values ?  

This is a common use case where we need to merge two columns into one row.


```python
df = df.with_columns(merged_cat = pl.struct(["group","score_bin"]))
df
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (7, 11)</small><table class="dataframe"><thead><tr><th>group</th><th>grade</th><th>score</th><th>date1</th><th>date2</th><th>rating</th><th>zone</th><th>class</th><th>mean_over_grp</th><th>score_bin</th><th>merged_cat</th></tr><tr><td>str</td><td>i16</td><td>str</td><td>date</td><td>date</td><td>f64</td><td>f64</td><td>str</td><td>i64</td><td>cat</td><td>struct[2]</td></tr></thead><tbody><tr><td>&quot;one&quot;</td><td>1</td><td>&quot;45&quot;</td><td>2023-01-01</td><td>2023-02-01</td><td>4.5</td><td>null</td><td>&quot;x1&quot;</td><td>170</td><td>&quot;1&quot;</td><td>{&quot;one&quot;,&quot;1&quot;}</td></tr><tr><td>&quot;one&quot;</td><td>98</td><td>&quot;49&quot;</td><td>2023-01-02</td><td>2023-02-02</td><td>3.5</td><td>NaN</td><td>null</td><td>170</td><td>&quot;3&quot;</td><td>{&quot;one&quot;,&quot;3&quot;}</td></tr><tr><td>&quot;one&quot;</td><td>2</td><td>&quot;76&quot;</td><td>2023-01-03</td><td>2023-02-03</td><td>4.0</td><td>1.0</td><td>null</td><td>170</td><td>&quot;5&quot;</td><td>{&quot;one&quot;,&quot;5&quot;}</td></tr><tr><td>&quot;two&quot;</td><td>3</td><td>&quot;83&quot;</td><td>2023-01-04</td><td>2023-02-04</td><td>1.0</td><td>2.0</td><td>&quot;c1&quot;</td><td>152</td><td>&quot;5&quot;</td><td>{&quot;two&quot;,&quot;5&quot;}</td></tr><tr><td>&quot;two&quot;</td><td>99</td><td>&quot;69&quot;</td><td>2023-01-05</td><td>2023-02-05</td><td>1.2</td><td>4.0</td><td>&quot;c2&quot;</td><td>152</td><td>&quot;1&quot;</td><td>{&quot;two&quot;,&quot;1&quot;}</td></tr><tr><td>&quot;three&quot;</td><td>23</td><td>&quot;90&quot;</td><td>2023-01-06</td><td>2023-02-06</td><td>9.4</td><td>5.0</td><td>&quot;x2&quot;</td><td>170</td><td>&quot;5&quot;</td><td>{&quot;three&quot;,&quot;5&quot;}</td></tr><tr><td>&quot;three&quot;</td><td>11</td><td>&quot;80&quot;</td><td>2023-01-07</td><td>2023-02-07</td><td>9.1</td><td>1.0</td><td>&quot;j1&quot;</td><td>170</td><td>&quot;1&quot;</td><td>{&quot;three&quot;,&quot;1&quot;}</td></tr></tbody></table></div>




```python
# lets check the first now at index 0
df.select(pl.col('merged_cat').take(0)).to_numpy()
```




    array([[{'group': 'one', 'score_bin': '1'}]], dtype=object)




```python
# let create a column with list as output
df=df.with_columns(
    merged_list = pl.struct(["group","class"]).apply(lambda x: [x['group'], x['class']])
)
df
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (7, 12)</small><table class="dataframe"><thead><tr><th>group</th><th>grade</th><th>score</th><th>date1</th><th>date2</th><th>rating</th><th>zone</th><th>class</th><th>mean_over_grp</th><th>score_bin</th><th>merged_cat</th><th>merged_list</th></tr><tr><td>str</td><td>i16</td><td>str</td><td>date</td><td>date</td><td>f64</td><td>f64</td><td>str</td><td>i64</td><td>cat</td><td>struct[2]</td><td>list[str]</td></tr></thead><tbody><tr><td>&quot;one&quot;</td><td>1</td><td>&quot;45&quot;</td><td>2023-01-01</td><td>2023-02-01</td><td>4.5</td><td>null</td><td>&quot;x1&quot;</td><td>170</td><td>&quot;1&quot;</td><td>{&quot;one&quot;,&quot;1&quot;}</td><td>[&quot;one&quot;, &quot;x1&quot;]</td></tr><tr><td>&quot;one&quot;</td><td>98</td><td>&quot;49&quot;</td><td>2023-01-02</td><td>2023-02-02</td><td>3.5</td><td>NaN</td><td>null</td><td>170</td><td>&quot;3&quot;</td><td>{&quot;one&quot;,&quot;3&quot;}</td><td>[&quot;one&quot;, null]</td></tr><tr><td>&quot;one&quot;</td><td>2</td><td>&quot;76&quot;</td><td>2023-01-03</td><td>2023-02-03</td><td>4.0</td><td>1.0</td><td>null</td><td>170</td><td>&quot;5&quot;</td><td>{&quot;one&quot;,&quot;5&quot;}</td><td>[&quot;one&quot;, null]</td></tr><tr><td>&quot;two&quot;</td><td>3</td><td>&quot;83&quot;</td><td>2023-01-04</td><td>2023-02-04</td><td>1.0</td><td>2.0</td><td>&quot;c1&quot;</td><td>152</td><td>&quot;5&quot;</td><td>{&quot;two&quot;,&quot;5&quot;}</td><td>[&quot;two&quot;, &quot;c1&quot;]</td></tr><tr><td>&quot;two&quot;</td><td>99</td><td>&quot;69&quot;</td><td>2023-01-05</td><td>2023-02-05</td><td>1.2</td><td>4.0</td><td>&quot;c2&quot;</td><td>152</td><td>&quot;1&quot;</td><td>{&quot;two&quot;,&quot;1&quot;}</td><td>[&quot;two&quot;, &quot;c2&quot;]</td></tr><tr><td>&quot;three&quot;</td><td>23</td><td>&quot;90&quot;</td><td>2023-01-06</td><td>2023-02-06</td><td>9.4</td><td>5.0</td><td>&quot;x2&quot;</td><td>170</td><td>&quot;5&quot;</td><td>{&quot;three&quot;,&quot;5&quot;}</td><td>[&quot;three&quot;, &quot;x2&quot;]</td></tr><tr><td>&quot;three&quot;</td><td>11</td><td>&quot;80&quot;</td><td>2023-01-07</td><td>2023-02-07</td><td>9.1</td><td>1.0</td><td>&quot;j1&quot;</td><td>170</td><td>&quot;1&quot;</td><td>{&quot;three&quot;,&quot;1&quot;}</td><td>[&quot;three&quot;, &quot;j1&quot;]</td></tr></tbody></table></div>





#### Q. How to convert dataframe columns into a python dict ?

Converting two columns into a key:value format.


```python
dict(df.select(pl.col(["score_bin", "score"])).iter_rows())
```




    {'1': '80', '3': '49', '5': '90'}





#### Q. How to convert a column containing list into separate rows? 

Similar to pandas, polars also provide `.explode` function to achieve this.


```python
df=pl.DataFrame(
    {
        "col1": [["X", "Y", "Z"], ["F", "G"], ["P"]],
        "col2": [["A", "B", "C"], ["C"], ["D", "E"]],
    }
).with_row_count()
df.explode(["col1"]).explode(["col2"]).head(4)
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (4, 3)</small><table class="dataframe"><thead><tr><th>row_nr</th><th>col1</th><th>col2</th></tr><tr><td>u32</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>0</td><td>&quot;X&quot;</td><td>&quot;A&quot;</td></tr><tr><td>0</td><td>&quot;X&quot;</td><td>&quot;B&quot;</td></tr><tr><td>0</td><td>&quot;X&quot;</td><td>&quot;C&quot;</td></tr><tr><td>0</td><td>&quot;Y&quot;</td><td>&quot;A&quot;</td></tr></tbody></table></div>





#### Q. How to perform string related functions using polars? 

Similar to pandas, polars also provide string methods using `<col_name>.str.<method_name>` format.


```python
# add prefix to columns except one
df.select(pl.all().exclude(['score']).suffix('_pre')).head(3)
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (3, 7)</small><table class="dataframe"><thead><tr><th>group_pre</th><th>grade_pre</th><th>date1_pre</th><th>date2_pre</th><th>rating_pre</th><th>zone_pre</th><th>class_pre</th></tr><tr><td>str</td><td>str</td><td>date</td><td>date</td><td>f64</td><td>f64</td><td>str</td></tr></thead><tbody><tr><td>&quot;one&quot;</td><td>&quot;1&quot;</td><td>2023-01-01</td><td>2023-02-01</td><td>4.5</td><td>null</td><td>&quot;x1&quot;</td></tr><tr><td>&quot;one&quot;</td><td>&quot;98&quot;</td><td>2023-01-02</td><td>2023-02-02</td><td>3.5</td><td>NaN</td><td>null</td></tr><tr><td>&quot;one&quot;</td><td>&quot;2&quot;</td><td>2023-01-03</td><td>2023-02-03</td><td>4.0</td><td>1.0</td><td>null</td></tr></tbody></table></div>




```python
# convert a column to uppercase
df.with_columns(pl.col('group').str.to_uppercase()).head()
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (5, 8)</small><table class="dataframe"><thead><tr><th>group</th><th>grade</th><th>score</th><th>date1</th><th>date2</th><th>rating</th><th>zone</th><th>class</th></tr><tr><td>str</td><td>str</td><td>i64</td><td>date</td><td>date</td><td>f64</td><td>f64</td><td>str</td></tr></thead><tbody><tr><td>&quot;ONE&quot;</td><td>&quot;1&quot;</td><td>45</td><td>2023-01-01</td><td>2023-02-01</td><td>4.5</td><td>null</td><td>&quot;x1&quot;</td></tr><tr><td>&quot;ONE&quot;</td><td>&quot;98&quot;</td><td>49</td><td>2023-01-02</td><td>2023-02-02</td><td>3.5</td><td>NaN</td><td>null</td></tr><tr><td>&quot;ONE&quot;</td><td>&quot;2&quot;</td><td>76</td><td>2023-01-03</td><td>2023-02-03</td><td>4.0</td><td>1.0</td><td>null</td></tr><tr><td>&quot;TWO&quot;</td><td>&quot;3&quot;</td><td>83</td><td>2023-01-04</td><td>2023-02-04</td><td>1.0</td><td>2.0</td><td>&quot;c1&quot;</td></tr><tr><td>&quot;TWO&quot;</td><td>&quot;99&quot;</td><td>69</td><td>2023-01-05</td><td>2023-02-05</td><td>1.2</td><td>4.0</td><td>&quot;c2&quot;</td></tr></tbody></table></div>




```python
# concatenate a lists into a string 
(df
.with_columns(
    merged_list = pl.struct(["group","class"]).apply(lambda x: [x['group'], x['class']])
).with_columns(
    m_con=pl.col("merged_list").arr.join("_"),
    v4=pl.col("merged_list").arr.concat(pl.col("merged_list")), # concatenate two columns having list into a bigger list
))
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (7, 11)</small><table class="dataframe"><thead><tr><th>group</th><th>grade</th><th>score</th><th>date1</th><th>date2</th><th>rating</th><th>zone</th><th>class</th><th>merged_list</th><th>m_con</th><th>v4</th></tr><tr><td>str</td><td>str</td><td>i64</td><td>date</td><td>date</td><td>f64</td><td>f64</td><td>str</td><td>list[str]</td><td>str</td><td>list[str]</td></tr></thead><tbody><tr><td>&quot;one&quot;</td><td>&quot;1&quot;</td><td>45</td><td>2023-01-01</td><td>2023-02-01</td><td>4.5</td><td>null</td><td>&quot;x1&quot;</td><td>[&quot;one&quot;, &quot;x1&quot;]</td><td>&quot;one_x1&quot;</td><td>[&quot;one&quot;, &quot;x1&quot;, … &quot;x1&quot;]</td></tr><tr><td>&quot;one&quot;</td><td>&quot;98&quot;</td><td>49</td><td>2023-01-02</td><td>2023-02-02</td><td>3.5</td><td>NaN</td><td>null</td><td>[&quot;one&quot;, null]</td><td>&quot;one_null&quot;</td><td>[&quot;one&quot;, null, … null]</td></tr><tr><td>&quot;one&quot;</td><td>&quot;2&quot;</td><td>76</td><td>2023-01-03</td><td>2023-02-03</td><td>4.0</td><td>1.0</td><td>null</td><td>[&quot;one&quot;, null]</td><td>&quot;one_null&quot;</td><td>[&quot;one&quot;, null, … null]</td></tr><tr><td>&quot;two&quot;</td><td>&quot;3&quot;</td><td>83</td><td>2023-01-04</td><td>2023-02-04</td><td>1.0</td><td>2.0</td><td>&quot;c1&quot;</td><td>[&quot;two&quot;, &quot;c1&quot;]</td><td>&quot;two_c1&quot;</td><td>[&quot;two&quot;, &quot;c1&quot;, … &quot;c1&quot;]</td></tr><tr><td>&quot;two&quot;</td><td>&quot;99&quot;</td><td>69</td><td>2023-01-05</td><td>2023-02-05</td><td>1.2</td><td>4.0</td><td>&quot;c2&quot;</td><td>[&quot;two&quot;, &quot;c2&quot;]</td><td>&quot;two_c2&quot;</td><td>[&quot;two&quot;, &quot;c2&quot;, … &quot;c2&quot;]</td></tr><tr><td>&quot;three&quot;</td><td>&quot;23&quot;</td><td>90</td><td>2023-01-06</td><td>2023-02-06</td><td>9.4</td><td>5.0</td><td>&quot;x2&quot;</td><td>[&quot;three&quot;, &quot;x2&quot;]</td><td>&quot;three_x2&quot;</td><td>[&quot;three&quot;, &quot;x2&quot;, … &quot;x2&quot;]</td></tr><tr><td>&quot;three&quot;</td><td>&quot;11&quot;</td><td>80</td><td>2023-01-07</td><td>2023-02-07</td><td>9.1</td><td>1.0</td><td>&quot;j1&quot;</td><td>[&quot;three&quot;, &quot;j1&quot;]</td><td>&quot;three_j1&quot;</td><td>[&quot;three&quot;, &quot;j1&quot;, … &quot;j1&quot;]</td></tr></tbody></table></div>





#### Q. How to perform datetime functions using polars? 

Similar to pandas, polars also provide datetime methods using `<col_name>.dt.<method_name>` format.


```python
df.with_columns(day = pl.col('date1').dt.day(),
                month=  pl.col('date1').dt.month(),
                year=  pl.col('date1').dt.year(),
                year_mon = pl.col('date1').cast(pl.Utf8).str.slice(0, 7),
               )
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (7, 12)</small><table class="dataframe"><thead><tr><th>group</th><th>grade</th><th>score</th><th>date1</th><th>date2</th><th>rating</th><th>zone</th><th>class</th><th>day</th><th>month</th><th>year</th><th>year_mon</th></tr><tr><td>str</td><td>str</td><td>i64</td><td>date</td><td>date</td><td>f64</td><td>f64</td><td>str</td><td>u32</td><td>u32</td><td>i32</td><td>str</td></tr></thead><tbody><tr><td>&quot;one&quot;</td><td>&quot;1&quot;</td><td>45</td><td>2023-01-01</td><td>2023-02-01</td><td>4.5</td><td>null</td><td>&quot;x1&quot;</td><td>1</td><td>1</td><td>2023</td><td>&quot;2023-01&quot;</td></tr><tr><td>&quot;one&quot;</td><td>&quot;98&quot;</td><td>49</td><td>2023-01-02</td><td>2023-02-02</td><td>3.5</td><td>NaN</td><td>null</td><td>2</td><td>1</td><td>2023</td><td>&quot;2023-01&quot;</td></tr><tr><td>&quot;one&quot;</td><td>&quot;2&quot;</td><td>76</td><td>2023-01-03</td><td>2023-02-03</td><td>4.0</td><td>1.0</td><td>null</td><td>3</td><td>1</td><td>2023</td><td>&quot;2023-01&quot;</td></tr><tr><td>&quot;two&quot;</td><td>&quot;3&quot;</td><td>83</td><td>2023-01-04</td><td>2023-02-04</td><td>1.0</td><td>2.0</td><td>&quot;c1&quot;</td><td>4</td><td>1</td><td>2023</td><td>&quot;2023-01&quot;</td></tr><tr><td>&quot;two&quot;</td><td>&quot;99&quot;</td><td>69</td><td>2023-01-05</td><td>2023-02-05</td><td>1.2</td><td>4.0</td><td>&quot;c2&quot;</td><td>5</td><td>1</td><td>2023</td><td>&quot;2023-01&quot;</td></tr><tr><td>&quot;three&quot;</td><td>&quot;23&quot;</td><td>90</td><td>2023-01-06</td><td>2023-02-06</td><td>9.4</td><td>5.0</td><td>&quot;x2&quot;</td><td>6</td><td>1</td><td>2023</td><td>&quot;2023-01&quot;</td></tr><tr><td>&quot;three&quot;</td><td>&quot;11&quot;</td><td>80</td><td>2023-01-07</td><td>2023-02-07</td><td>9.1</td><td>1.0</td><td>&quot;j1&quot;</td><td>7</td><td>1</td><td>2023</td><td>&quot;2023-01&quot;</td></tr></tbody></table></div>





#### Q. How to read parquet data from S3 with polars? 

I couldn't find a straight forward way to do this, hence wrote a utils functions which can read multiple parquet files from a S3 directory.


```python
def read_with_polars(s3_path: str):
    bucket, _, key = s3_path.replace("s3://", "").partition("/")
    s3 = boto3.resource("s3")
    s3bucket = s3.Bucket(bucket)
    paths = [
        f"s3://{bucket}/{obj.key}"
        for obj in s3bucket.objects.filter(Prefix=key)
    ]
    print(len(paths))
    return pl.concat(
        [
            pl.read_parquet(path, use_pyarrow=True, memory_map=True)
            for path in paths
        ]
    )
df = read_with_polars('s3://data-science-dev/polars/data/')
```



### Summary

Overall, my experience using polars has been quite satisfying. Having been deployed a python project using polars I feel more confident about using it while handling large data sets. I would probably continue using pandas as long as it doesn't slow down significantly. Also, I tried building a xgboost model on polars dataframes, it worked nicely without any need to convert it to pandas dataframe. 


