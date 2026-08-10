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
