Here are some things to note to understand the diagram better:
In the first plot each dot represents one face image.
Colors represent different people (Person ID 0–39).
Proximity means similarity: Dots close together are faces that UMAP considers similar in the 4096-dimensional space
Though it doesn’t happen in real-world scenarios, ideally, we would expect all 10 images of the same person (same color) to cluster together. 
In the image, we see some of the similar colors clustered together when we built the UMAP using the default parameters.

In the second set of graphs the grid illustrates how parameters impact the visualization. Considering the top row of n_neighbors effect:
    1) n_neighbors=5: Very local view, creates many small, tight clusters.
    2) n_neighbors=15: Balanced view, gives a good mix of local and global structure.
    3) n_neighbors=50: Global view, it focuses on the overall structure, potentially merging similar individuals.

Considering the bottom row of min_dist effect:
    1) min_dist=0.0: Points are extremely close, as seen in the grid. This creates very tight, dense clusters.
    2) min_dist=0.5: Medium spacing resulting in clusters that are more spread out.
    3) min_dist=0.99: Maximum spacing results in points being evenly distributed; however loses cluster structure.

It’s always advisable to experiment with these parameters to get the best clustering result for the problem at hand.

The PCA projection (left) shows a dense cloud of points centered around the origin with extensive overlap between different colored dots. This is exactly what we expect from PCA — it captures the directions of maximum variance in the data, but fails to separate the different people effectively. We can see that faces from different individuals (different colors) are completely mixed together, making it nearly impossible to identify distinct groups.

The t-SNE projection (middle) shows improvement with well-separated clusters. Each color (representing a different person) forms a distinct, compact group with clear boundaries between different individuals. Notice how t-SNE creates almost perfect local groupings where faces from the same person are pulled very close together while being pushed away from other groups. This is t-SNE’s strength: it excels at preserving local neighborhoods and creating visually distinct clusters. 

The UMAP projection (right) takes a different approach. While the clusters are more spread out compared to t-SNE, this reflects UMAP’s attempt to preserve both local and global structure. We can see that each person still forms recognizable groups, but with more internal variation. Some clusters show slight overlap or are positioned closer together, which might indicate actual similarities between those individuals.
