#include <vector>
#include <math.h>
#include <cfloat>
#include <bits/stdc++.h>
#include <algorithm>
#include <execution>
#include <stack>
#include <queue>
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

class fofW {
    public:
        fofW(std::vector<float> X_locations, std::vector<float> Y_locations, std::vector<float> SNR, std::vector<float> Widths);

        void do_clustering();

        std::vector<std::vector<size_t>> final_clusters();

    private:
        std::vector<float> _widths, _x, _y, _snr;
        std::vector<std::vector<size_t>> _small_clusters;
        std::vector<std::vector<size_t>> _final_clusters;
        std::vector<size_t> _representative_elements;

        float distance(size_t idx1, size_t idx2);
        std::vector<size_t> _connections;
        std::vector<size_t> _status;
	float _matching_factor;
};
