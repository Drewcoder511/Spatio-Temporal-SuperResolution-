# Spatio-Temporal-SuperResolution-
GNNs with temporal data

The task is interpolation (for node features) on Graph data.
I use my customized time-aware random walk method to traverse the graph data arranged in a specific time sequence and obtain the graph representation. I then utilize GNN to predict the node features on the graph at a specific time point.

For example: On AUS Weather dataset, with 49 nodes which represent cities in Aus and 1577 timestamps which represent each day, our goal is to predict the node features on time=2, 4, 6...of each node (city), supposing we only have data on time=1, 3, 5...

We have two dataset for this experiment.

AUS Weather:

AUS Weather dataset contains weather information for 49 locations in Australia, represented as nodes in the graph. The dataset covers a time period of three years, from January 1st, 2016 to December 31st, 2018, and provides weather information at an hourly time interval.

Therefore, the dataset includes approximately 26,280 data points per location (24 hours x 365 days x 3 years) and a total of approximately 1,287,720 data points (26,280 x 49 locations) across the entire dataset.

The graph structure is constructed based on the proximity of weather stations to each other, resulting in a weighted graph with varying degrees of connectivity between nodes. The AUS Weather dataset is a valuable resource for studying spatial-temporal relationships between weather variables in Australia and developing weather prediction models.


Windmill large:

The Windmill Large dataset is a graph-based dataset that represents the power generation of wind turbines located in a wind farm. The dataset includes 200 nodes, where each node represents a wind turbine, and it covers a time period of one year, from January 1st, 2019 to December 31st, 2019, providing power generation data at a one-hour time interval.

Therefore, the Windmill Large dataset includes approximately 17,520 data points (24 hours x 365 days) per node, and a total of approximately 5,584,880 data points (17,520 x 319 nodes) across the entire dataset.

Each node in the graph represents a wind turbine, and the edges between nodes are created based on the physical proximity of the wind turbines within the wind farm. The node features represent the power generation of each wind turbine at a given time, measured in kilowatts. The dataset also includes additional features such as temperature and wind speed, which are measured at the location of each wind turbine and can be used to enhance predictive models.

The Windmill Large dataset is a valuable resource for studying the performance of wind turbines in a wind farm, optimizing power generation, and developing predictive models for power generation.

Some files explained:
RandomWalk02: Including my designed time-aware random walk methods and some other basic models definition.

