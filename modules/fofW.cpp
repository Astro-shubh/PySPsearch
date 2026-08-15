#include "fofW.h"


fofW::fofW(std::vector<float> X_locations, std::vector<float> Y_locations, std::vector<float> snr, std::vector<float> widths)
    :_widths(widths),
    _x(X_locations),
    _y(Y_locations),
    _snr(snr)
{
}


void fofW::do_clustering() {
// Point defines a boost geometry model for 2D cartesian point
    typedef bg::model::point<float, 2, bg::cs::cartesian> Point;
// RTreeValue is the data type used by rtree to make the tree, it's pair with point and index
    typedef std::pair<Point, size_t> RTreeValue;
// Do nothing if input is emplty
    size_t n = _x.size();
    if (n == 0) return;

// A boolean vector to store the status of index, it tells if the point at index has been visited or not
    std::vector<bool> visited(n, false);
// Make sure final cluster is empty
    _final_clusters.clear();

// Store the rtree points in a vector
    std::vector<RTreeValue> rtree_entries;
    for (size_t i = 0; i < n; ++i)
    {
        rtree_entries.push_back(std::make_pair(Point(_x[i], _y[i]), i));
    }
// Make a linear rtree with max box of 16 children
    bgi::rtree<RTreeValue, bgi::linear<16>> rtree(rtree_entries);


    while (!rtree.empty())
    {
// current_final_cluster stores the indices connected together by this iteration of while loop
        std::vector<size_t> current_final_cluster;
// traversal_queue stores the connected set of neighbours that are not part of cluster yet
        std::queue<RTreeValue> traversal_queue;
// Get the first member of the first box of top node
	RTreeValue seed = *rtree.qbegin(bgi::satisfies([](RTreeValue const&){ return true; }));
// remove from the tree as it is going to be clustered now
        rtree.remove(seed);
// This is the first point to form the neighbourhood
        traversal_queue.push(seed);


        while (!traversal_queue.empty())
	{
	    // Get the first member from the traversal_queue
            RTreeValue curr_node = traversal_queue.front();
	    // remove this from the queue as it going to be clustered
            traversal_queue.pop();
	    // current index and point
            size_t curr_idx = curr_node.second;
            Point curr_pt = curr_node.first;
	    // put the index in current cluster
            current_final_cluster.push_back(curr_idx);
            // Get the linking length
            float current_W = _widths[curr_idx];
            float current_W_sq = current_W * current_W; // For faster distance check

            // Define search box
            bg::model::box<Point> box(
                Point(curr_pt.get<0>() - current_W, curr_pt.get<1>() - current_W),
                Point(curr_pt.get<0>() + current_W, curr_pt.get<1>() + current_W)
            );

            // Query and collect neighbors
            std::vector<RTreeValue> neighbors;
            rtree.query(bgi::within(box) && bgi::satisfies([&](RTreeValue const& v) {
                // Precise distance check using squared values (no sqrt)
                float dx = curr_pt.get<0>() - v.first.get<0>();
                float dy = curr_pt.get<1>() - v.first.get<1>();
                return (dx*dx + dy*dy) < current_W_sq;
            }), std::back_inserter(neighbors));

            // 4. REMOVE neighbors from tree so they aren't found again
            if (!neighbors.empty())
	    {
                rtree.remove(neighbors.begin(), neighbors.end());
                for (const auto& neighbor : neighbors)
		{
                    traversal_queue.push(neighbor);
                }
            }
        }
            // Add points to final output
        _final_clusters.push_back(current_final_cluster);
    }

}

std::vector<std::vector<std::size_t>> fofW::final_clusters()
{
    return _final_clusters;
}

