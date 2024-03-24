# Data Format Guidelines for EpiNet Analysis

## Overview

This document outlines the data format requirements for the EpiNet Analysis project. To ensure compatibility and optimal analysis performance, please format your epidemiological data according to the specifications detailed below.

## Data Structure

The EpiNet Analysis project processes data structured as a network of nodes and edges, where nodes represent entities (e.g., individuals, locations, risk factors) and edges represent relationships or interactions between these entities.

### Nodes

Each node in the network should be represented by a unique identifier and can have multiple attributes associated with it. The following attributes are required for each node:

- **ID**: A unique identifier (integer or string).
- **Type**: The type of entity (e.g., "Individual", "Location").
- Additional attributes relevant to the analysis (e.g., age, sex, health status for individuals).

Node data should be provided in a CSV file with the following format:

```
ID,Type,Attribute1,Attribute2,...
1,Individual,25,Male,...
2,Location,Downtown,...
```

### Edges

Edges represent the relationships or interactions between nodes. Each edge should specify the IDs of the two nodes it connects, along with any attributes relevant to that connection (e.g., relationship type, interaction strength).

Edge data should be provided in a CSV file with the following format:

```
SourceID,TargetID,Attribute1,Attribute2,...
1,2,Friend,High,...
```

## File Format

- All data should be provided in CSV (Comma-Separated Values) format.
- UTF-8 encoding is recommended to support a wide range of characters.

## Example

Here's a simple example illustrating the expected data format:

**nodes.csv**
```
ID,Type,Age,Sex
1,Individual,25,Male
2,Individual,30,Female
3,Location,CityCenter
```

**edges.csv**
```
SourceID,TargetID,Relationship,InteractionStrength
1,2,Friend,High
1,3,Visits,Daily
2,3,Visits,Weekly
```

## Tips for Data Preparation

- Ensure that IDs are consistent across the node and edge files.
- Remove any duplicate or irrelevant entries before submission.
- Verify that all required attributes are included for each node and edge.

By following these data format guidelines, you will help streamline the data ingestion and analysis process, enabling more accurate and efficient outcomes from the EpiNet Analysis project.