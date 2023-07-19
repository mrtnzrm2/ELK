#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <algorithm>
#include <numeric> 
#include <cmath>
#include "hclust-cpp/fastcluster.h"
#include "entropy_tools.cpp"

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

struct lcprops {
  int m = 0;
  int n = 0;
};

double sum(std::vector<double> &v) {
  if (v.size() == 0) {
    printf("Warning: mean of vector with zero size\n");
    return 0;
  }
  double mv = 0;
  for (int i = 0; i < v.size(); i++)
    mv += v[i];
  return mv;
}

template<typename T>
void unique(std::vector<T> &v) {
  std::vector<int>::iterator ip;
  std::sort(v.begin(), v.end());
  ip = std::unique(
    v.begin(),
    v.begin() + v.size()
  );
  v.resize(
    std::distance(v.begin(), ip)
  );
}

template<typename T>
void intersection(
  std::vector<T> &a, std::vector<T> &b, std::vector<T> &result
) {
  std::set_intersection(
    a.begin(), a.end(),
    b.begin(), b.end(),
    back_inserter(result)
  );
}

bool search_key(std::map<int, std::vector<int> > &a, const int &key) {
  for (std::map<int, std::vector<int> >::iterator f=a.begin(); f != a.end(); ++f) {
    if (f->first == key) return true;
  }
  return false;
}

bool search_key(std::map<int, lcprops > &a, const int &key) {
  for (std::map<int, lcprops>::iterator f=a.begin(); f != a.end(); ++f) {
    if (f->first == key) return true;
  }
  return false;
}

std::vector<double> simplify_height_to_k_end(
  int &n,
  double* height,
  std::vector<double>& sim_height,
  int  &size
) {
  double h = height[0];
  std::vector<double> sim_k;
  for (int i = 0; i < n - 1; i++) {
    if (i < n - 2) {
      if (height[i + 1] != h) {
        sim_k.push_back(n - i);
        sim_height.push_back(h);
        h = height[i + 1];
        ++(size);
      }
    } else {
      if (height[i] != height[i - 1]) {
        sim_k.push_back(n - i);
        h = height[i];
        sim_height.push_back(h);
        ++(size);
      }
    }
    
  }
  return sim_k;
}

std::vector<double> simplify_height_to_k_start(
  int &n,
  double* height,
  std::vector<double>& sim_height,
  int &size
) {
  double h = height[0];
  std::vector<double> sim_k;
  for (int i = 0; i < n - 1; i++) {
    if (i == 0) {
      sim_k.push_back(n - 1);
      sim_height.push_back(h);
      ++size;
    }
    if (height[i] != h && i != 0) {
      h = height[i];
      sim_k.push_back(n - i - 1);
      sim_height.push_back(h);
      ++size;
    }
  }
  return sim_k;
}

std::vector<double> complete_height_to_k(
  int &n,
  double* height,
  std::vector<double>& sim_height,
  int &size
) {
  std::vector<double> sim_k;
  for (int i = 0; i < n - 1; i++) {
    sim_k.push_back(n - i - 1);
    sim_height.push_back(height[i]);
    size++;
}
  return sim_k;
}

double Dc(int &m, int &n, bool &undirected) {
  double dc;
  if (!undirected)
    dc = (m - n + 1.) / pow(n - 1., 2.);
  else
    dc = (m - n + 1.) / ((n * (n - 1.) / 2.) - n + 1.);
  if (dc > 0) return dc;
  else return 0;
}

double Sc(int &m, int &n, int &M, int& N) {
  double pc;
  pc = (m - n + 1.) / (M - N + 1.);
  if (pc > 0) return -pc * log(pc);
  else return 0;
}

double Xsus(std::map<int, lcprops> &v, int &N, int &order) {
  double  x = 0; // percolation suceptability
  for (std::map<int, lcprops >::iterator it = v.begin(); it != v.end(); ++it) {
    if (it->second.m <= 1 || it->second.n <= 2) continue;
    if (it->second.m != order)
      x += pow(it->second.m, 2.);
  }
  return x / N;
}

double order_parameter(std::vector<int> &v, int &M) {
  return v[0] / (M * 1.);
}

double Xm(std::map<int, lcprops> &v) {
  double n = v.size();
  double xm2 = 0, xm = 0;
  std::map<int, int> v_count;
  for (std::map<int, lcprops>::iterator it = v.begin(); it != v.end(); ++it) {
    if (!search_key(v, it->second.m)) v_count[it->second.m] = 1;
    else v_count[it->second.m]++;
  }
  for (std::map<int, int >::iterator it = v_count.begin(); it != v_count.end(); ++it) {
    xm2 += pow(it->first, 2.0) * it->second;
    xm += it->first * it->second * 1.;
  }
  if (xm > 0)
    return xm2 / xm;
  else return 0;
}

class ph {
  private:

    std::vector<int> K;
    std::vector<double> Height;
    std::vector<int> NEC;
    std::vector<double> D;
    std::vector<int> ntrees;
    std::vector<double> X;
    std::vector<double> OrP;
    std::vector<double> XM;
    std::vector<double> S;

    std::vector<double> Sh;
    std::vector<double> Sv;
    std::vector<double> ShH;
    std::vector<double> SvH;
    int max_level=0;

    int number_of_elements;
    std::vector<double> distane_matrix;
    std::vector<int> source_vertices;
    std::vector<int> target_vertices;
    int total_nodes;
    int Linkage;
    bool CUT;

    bool undirected;

  public:
    ph(
      const int n,
      std::vector<double> distmat,
      std::vector<int> source,
      std::vector<int> target,
      const int nodes,
      const int linkage,
      const bool cut,
      const bool uni
    );
    ~ph(){};

    void vite();
    void vite_nodewise(std::vector<int> &equivalence, std::vector<double> &h, int &array_size);
    void arbre(std::string &t_size);

    template <typename T>
    void expand_vector(std::vector<T>& v, const int& N);

    std::vector<int> get_K();
    std::vector<double> get_Height();
    std::vector<int> get_NEC();
    std::vector<double> get_D();
    std::vector<int> get_ntrees();
    std::vector<double> get_X();
    std::vector<double> get_OrP();
    std::vector<double> get_XM();
    std::vector<double> get_S();
    std::vector<double> get_entropy_h();
    std::vector<double> get_entropy_v();
    std::vector<double> get_entropy_h_H();
    std::vector<double> get_entropy_v_H();
    int get_max_level();
    void get_sizes(
      std::map<int, lcprops> &info_sizes,
      int* labels, std::vector<int> &lcsize,
      std::vector<int>& unique_labels,
      std::vector<int>& source,
      std::vector<int>& target,
      int& n
    );
};

ph::ph(
  const int n,
  std::vector<double> distmat,
  std::vector<int> source,
  std::vector<int> target,
  const int nodes,
  const int linkage,
  const bool cut,
  const bool uni
) {
  number_of_elements = n;
  distane_matrix = distmat;
  source_vertices = source;
  target_vertices = target;
  total_nodes = nodes;
  Linkage = linkage;
  CUT = cut;
  undirected = uni;
}

template <typename T>
void ph::expand_vector(std::vector<T>& v, const int& N) {
  v = std::vector<T>(N, 0);
}

std::vector<int> ph::get_K() {
  return K;
}
std::vector<double> ph::get_Height() {
  return Height;
}
std::vector<int> ph::get_NEC() {
  return NEC;
}

std::vector<double> ph::get_D() {
  return D;
}
std::vector<int> ph::get_ntrees() {
  return ntrees;
}
std::vector<double> ph::get_X() {
  return X;
}
std::vector<double> ph::get_OrP() {
  return OrP;
}
std::vector<double> ph::get_XM() {
  return XM;
}

std::vector<double> ph::get_S() {
  return S;
}

std::vector<double> ph::get_entropy_h(){
  return Sh;
}

std::vector<double> ph::get_entropy_v(){
  return Sv;
}

std::vector<double> ph::get_entropy_h_H(){
  return ShH;
}

std::vector<double> ph::get_entropy_v_H(){
  return SvH;
}

int ph::get_max_level(){
  return max_level;
}

void ph::arbre(std::string &t_size) {
  const std::string root = "L00";
  std::vector<double> H(number_of_elements, 0);
  expand_vector(Sh, number_of_elements);
  expand_vector(Sv, number_of_elements);
  expand_vector(ShH, number_of_elements);
  expand_vector(SvH, number_of_elements);
  std::vector<std::vector<int> > link_communities(number_of_elements, std::vector<int>(number_of_elements, 0));
  // Get hierarchy!! ----
  double* tri_distmat = new double[(number_of_elements * (number_of_elements - 1)) / 2];
  int* merge = new int[2 * (number_of_elements - 1)];
  double* height = new double[number_of_elements-1];
  int* labels = new int[number_of_elements];
  for (int i = 0; i < distane_matrix.size(); i++)
    tri_distmat[i] = distane_matrix[i];
  hclust_fast(
    number_of_elements,
    tri_distmat,
    Linkage,
    merge,
    height
  );
  // Get link community matrix ----
  for (int i=1; i <= number_of_elements - 1; i++) {
    cutree_k(number_of_elements, merge, i, labels);
    for (int j=0; j < number_of_elements; j++) link_communities[i-1][j] = labels[j];
  }
  for (int i=0; i < number_of_elements; i++) {
    link_communities[number_of_elements - 1][i] = i;
  }
  // Get heights ----
  for (int i=0; i < number_of_elements - 1; i++)
    H[i+1] = height[i];

  std::map<int, level_properties> chain;
  std::cout << "Starting Z2dict\n";
  std::map<std::string, vertex_properties> tree;
  Z2dict(link_communities, tree, H, t_size);

  std::cout << "Level information\n";
  level_information(tree, root, chain);

  // Get max level ----
  for (std::map<int, level_properties>::iterator it = chain.begin(); it != chain.end(); ++it) {
    if (it->first > max_level)
     max_level = it->first;
  }

  std::cout << "Vertex entropy\n";
  vertex_entropy(tree, chain, root, number_of_elements, Sh);
  std::cout << "Vertex entropy H\n";
  vertex_entropy_H(tree, chain, root, number_of_elements, max_level, ShH);
  std::cout << "Level entropy\n";
  level_entropy(tree, chain, root, number_of_elements, Sv);
  std::cout << "Level entropy H\n";
  level_entrop_H(tree, chain, root, number_of_elements, SvH);

  // Delete pointers
  delete[] labels;
  delete[] merge;
  delete[] height;
  delete[] tri_distmat;
}

void ph::get_sizes(
  std::map<int, lcprops> &info_sizes,
  int* labels, std::vector<int> &lcsize,
  std::vector<int>& unique_labels,
  std::vector<int>& source,
  std::vector<int>& target,
  int& n
) {
  std::vector<std::set<int> > node_buffer(unique_labels.size());
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < unique_labels.size(); j++) {
      if (labels[i] == unique_labels[j]) {
        info_sizes[unique_labels[j]].m++;
        lcsize[j]++;
        node_buffer[j].insert(source[i]);
        node_buffer[j].insert(target[i]);
      }
    }
  }
  for (int j = 0; j < unique_labels.size(); j++)
    info_sizes[unique_labels[j]].n = node_buffer[j].size();
  sort(lcsize.begin(), lcsize.end(), std::greater<int>());
}

void ph::vite() {
  // Various variables ----
  int nt, nec, length = 0;
  double dc, mtree;
  std::vector<double> sim_k, sim_height, dcv, scv;
  std::vector<int> lcsizes;
  std::map<int, lcprops > sizes;
  // Condense distance matrix ----
  double* tri_distmat = new double[(number_of_elements * (number_of_elements - 1)) / 2];
  // hclust arrays ----
  int* merge = new int[2 * (number_of_elements - 1)];
  double* height = new double[number_of_elements-1];
  int* labels = new int[number_of_elements];
  // Get condense matrix ----
  for (int i = 0; i < distane_matrix.size(); i++) {
    tri_distmat[i] = distane_matrix[i];
  }
  // Get hierarchy!! ----
  hclust_fast(
    number_of_elements,
    tri_distmat,
    Linkage, // linkage method
    merge,
    height
  );
  if (CUT) {
    // Delete duplicated heights preserving the first K and height ----
    sim_k = simplify_height_to_k_start(number_of_elements, height, sim_height, length);
  } else {
    // Keep the all the steps ----
    sim_k = complete_height_to_k(number_of_elements, height, sim_height, length);
  }
  expand_vector(K, length);
  expand_vector(Height, length);
  expand_vector(D, length);
  expand_vector(NEC, length);
  expand_vector(ntrees, length);
  expand_vector(X, length);
  expand_vector(XM, length);
  expand_vector(OrP, length);
  expand_vector(S, length);
  // THE GAME STARTS
  for (int i=0; i < length; i++) {
    K[i] = sim_k[i];
    Height[i] = sim_height[i];
    // Cut tree at given sim_k and get memberships ----
    cutree_k(
      number_of_elements,
      merge,
      sim_k[i],
      labels
    );
    std::vector<int> unique_labels(labels, labels + number_of_elements);
    unique(unique_labels);
    lcsizes = std::vector<int>(unique_labels.size(), 0);
    // Get number of links and number of nodes in link communities in order
    get_sizes(sizes, labels, lcsizes, unique_labels, source_vertices, target_vertices, number_of_elements);
    mtree = 0.;
    nec = 0;
    nt = 0;
    dcv = std::vector<double>(sizes.size(), 0.);
    scv = std::vector<double>(sizes.size(), 0.);
    for (std::map<int, lcprops >::iterator it=sizes.begin(); it != sizes.end(); ++it) {
      if (it->second.m > 1 && it->second.n > 2) {
        dcv[nec] = Dc(it->second.m, it->second.n , undirected) * it->second.m / number_of_elements;
        if (dcv[nec] <= 0) {
          nt++;
        }
        scv[nec] = Sc(it->second.m, it->second.n, number_of_elements, total_nodes);
        mtree += it->second.m - it->second.n + 1;
        nec++;
      }
    }
    S[i] = sum(scv);
    mtree = (number_of_elements - total_nodes + 1.) - mtree;
    mtree = mtree / (number_of_elements  - total_nodes + 1.);
    if (mtree > 0) S[i] += -mtree * log(mtree);
    D[i] = sum(dcv);
    ntrees[i] = nt;
    // NEC: number of edge compact LCs
    NEC[i] = nec;
    OrP[i] = order_parameter(lcsizes, number_of_elements);
    XM[i] = Xm(sizes);
    X[i] = Xsus(sizes, number_of_elements, lcsizes[0]);
    sizes.clear();
  }
  // Delete pointers
  delete[] labels;
  delete[] merge;
  delete[] height;
  delete[] tri_distmat;
}

void ph::vite_nodewise(std::vector<int> &equivalence, std::vector<double> &h, int &array_size) {
  // Various variables ----
  int nt, nec;
  double mtree;
  std::vector<double> dcv, scv;
  std::vector<int> lcsizes;
  std::map<int, lcprops > sizes;
  // Condense distance matrix ----
  double* tri_distmat = new double[(number_of_elements * (number_of_elements - 1)) / 2];
  // hclust arrays ----
  int* merge = new int[2 * (number_of_elements - 1)];
  double* height = new double[number_of_elements-1];
  int* labels = new int[number_of_elements];
  // Get condense matrix ----
  for (int i = 0; i < distane_matrix.size(); i++)
    tri_distmat[i] = distane_matrix[i];
  // Get hierarchy!! ----
  hclust_fast(
    number_of_elements,
    tri_distmat,
    Linkage, // linkage method
    merge,
    height
  );
  expand_vector(K, array_size);
  expand_vector(Height, array_size);
  expand_vector(D, array_size);
  expand_vector(NEC, array_size);
  expand_vector(ntrees, array_size);
  expand_vector(X, array_size);
  expand_vector(XM, array_size);
  expand_vector(OrP, array_size);
  expand_vector(S, array_size);
  // THE GAME STARTS
  for (int i=0; i < array_size; i++) {
    K[i] = equivalence[i];
    Height[i] = h[i];
    // Cut tree at given sim_k and get memberships ----
    cutree_k(
      number_of_elements,
      merge,
      K[i],
      labels
    );
    std::vector<int> unique_labels(labels, labels + number_of_elements);
    unique(unique_labels);
    lcsizes = std::vector<int>(unique_labels.size(), 0);
    // Get number of links and number of nodes in link communities in order
    get_sizes(sizes, labels, lcsizes, unique_labels, source_vertices, target_vertices, number_of_elements);
    mtree = 0.;
    nec = 0;
    nt = 0;
    dcv = std::vector<double>(sizes.size(), 0.);
    scv = std::vector<double>(sizes.size(), 0.);
    for (std::map<int, lcprops >::iterator it=sizes.begin(); it != sizes.end(); ++it) {
      if (it->second.m > 1 && it->second.n > 2) {
        dcv[nec] = Dc(it->second.m, it->second.n , undirected) * it->second.m / number_of_elements;
        if (dcv[nec] <= 0) {
          nt++;
        }
        scv[nec] = Sc(it->second.m, it->second.n, number_of_elements, total_nodes);
        mtree += it->second.m - it->second.n + 1;
        nec++;
      }
    }
    S[i] = sum(scv);
    mtree = number_of_elements * 1. - total_nodes + 1 - mtree;
    mtree = mtree / (number_of_elements * 1. - total_nodes + 1);
    if (mtree > 0) S[i] += -mtree * log(mtree);
    D[i] = sum(dcv);
    ntrees[i] = nt;
    // NEC: number of edge compact LCs
    NEC[i] = nec;
    OrP[i] = order_parameter(lcsizes, number_of_elements);
    XM[i] = Xm(sizes);
    X[i] = Xsus(sizes, number_of_elements, lcsizes[0]);
    sizes.clear();
  }
  // Delete pointers
  delete[] labels;
  delete[] merge;
  delete[] height;
  delete[] tri_distmat;
}

PYBIND11_MODULE(process_hclust, m) {
    py::class_<ph>(m, "ph")
        .def(
          py::init<
            const int,
            std::vector<double>,
            std::vector<int>,
            std::vector<int>,
            const int,
            const int,
            const bool,
            const bool
          >()
        )
        .def("vite", &ph::vite)
        .def("vite_nodewise", &ph::vite_nodewise)
        .def("arbre", &ph::arbre)
        .def("get_K", &ph::get_K)
				.def("get_Height", &ph::get_Height)
				.def("get_NEC", &ph::get_NEC)
				.def("get_D", &ph::get_D)
			  .def("get_ntrees", &ph::get_ntrees)
        .def("get_X", &ph::get_X)
        .def("get_OrP", &ph::get_OrP)
			  .def("get_XM", &ph::get_XM)
        .def("get_S", &ph::get_S)
        .def("get_entropy_h", &ph::get_entropy_h)
        .def("get_entropy_v", &ph::get_entropy_v)
        .def("get_entropy_h_H", &ph::get_entropy_h_H)
        .def("get_entropy_v_H", &ph::get_entropy_v_H)
        .def("get_max_level", &ph::get_max_level);
}