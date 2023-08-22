#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <stdio.h>
#include "hclust-cpp/fastcluster.h"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

using namespace pybind11::literals; 
namespace py = pybind11;

std::vector<int> NEC_one(std::vector<int>& NEC, int& n) {
  std::vector<int> buffer;
  for (int i = 0; i < n; i++) {
    if (NEC[i] >= 1)
      buffer.push_back(i);
  }
  return buffer;
}

std::vector<int> DC_valid(std::vector<double>& v) {
  std::vector<int> valid;
  for (int i = 0; i < v.size(); i++) {
    if (v[i] > 0) {
      valid.push_back(i);
    }
  }
  return valid;
}

void unique(std::vector<int>& v) {
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

std::vector<int> intersect(
  std::vector<int>& v,
  std::vector<int>& u
) {
  std::vector<int> intersect;
  std::set_intersection(
    v.begin(), v.end(),
    u.begin(), u.end(),
    back_inserter(intersect)
  );
  return intersect;
}

std::vector<double> link_communitiy_Dc(
  int* labels, int& leaves,
  std::vector<int>& source, std::vector<int>& target, bool &undirected
) {
  int edg, N;
  double dc;
  std::vector<int> nodes_src;
  std::vector<int> nodes_tgt;
  std::vector<int> lc_nodes;
  std::vector<int> vector_labels(labels, labels + leaves);
  unique(vector_labels);
  std::vector<double> Dc(vector_labels.size());
  for (int i = 0; i < vector_labels.size(); i++) {
    edg = 0;
    for (int j = 0; j < leaves; j++) {
      if ( labels[j] == vector_labels[i]) {
        nodes_src.push_back(source[j]);
        nodes_tgt.push_back(target[j]);
        edg++;
      }
    }
    lc_nodes = nodes_src;
    lc_nodes.insert(
      lc_nodes.end(), nodes_tgt.begin(), nodes_tgt.end()
    );
    unique(lc_nodes);
    N = lc_nodes.size();
    if (!undirected)
      dc = (edg - N + 1.) / (pow(N - 1., 2.0));
    else
      dc = (edg - N + 1.) / (N * (N - 1.) / 2. - N + 1.);
    Dc[i] = dc;
    nodes_src.clear();
    nodes_tgt.clear();
  }
  return Dc;
}

struct node {
  int merge;
  std::vector<int> members;
};

node create_node(int& t, std::vector<int>& v1, std::vector<int>& v2) {
  // Merge new members
  std::vector<int> new_members;
  new_members.reserve(
    v1.size() +
    v2.size()
  );
  new_members.insert(
    new_members.end(),
    v1.begin(),
    v2.end()
  );
  new_members.insert(
    new_members.end(),
    v1.begin(),
    v2.end()
  );
  unique(new_members);
  node new_node = {
    t + 1,
    new_members
  };
  return new_node;
}

template<typename T>
void print_v(std::vector<T>& v) {
  for (int i = 0; i < v.size(); i++) {
    std::cout << v[i] << " ";
  }
  std::cout << std::endl;
}

class noeud_arbre {

  private:
    // vite
    std::vector<double> distance;
    std::vector<int> source;
    std::vector<int> target;
    std::vector<int> K;
    std::vector<double> height_feature;
    std::vector<int> NEC;
    int nodes, leaves, linkage, n;
    bool undirected;
    // return
    std::vector<std::vector<double> > node_hierarchy;
    std::vector<std::vector<int> > equivalence;
  public:
    noeud_arbre(
      std::vector<double> distance_link_matrix,
      std::vector<int> source_nodes_edgelist,
      std::vector<int> target_nodes_edgelist,
      std::vector<int> number_link_partitions,
      std::vector<double> K_height,
      std::vector<int> number_non_trees,
      const int N,
      const int M,
      const int linkage_method,
      const int edege_list_size,
      const int sp,
      const bool uni
    ) {
      // Set class attributes
      distance = distance_link_matrix;
      source = source_nodes_edgelist;
      target = target_nodes_edgelist;
      K = number_link_partitions;
      height_feature = K_height;
      NEC = number_non_trees;
      nodes = N;
      leaves = M;
      linkage = linkage_method;
      n = edege_list_size;
      undirected = uni;
      // Initialize node hierarchy;
      for (int i = 0; i < nodes - 1; i++)
        node_hierarchy.push_back(std::vector<double>(4, 0.));
      vite(sp);
    }
    ~noeud_arbre(){};
    void vite(const int &sp);
    std::vector<std::vector<double> > get_node_hierarchy();
    std::vector<std::vector<int> > get_equivalence();
    void display_progress(int &ini,  const int &total, int &carrier, const int &sp);
};

std::vector<std::vector<double> > noeud_arbre::get_node_hierarchy() {
  return node_hierarchy;
}

std::vector<std::vector<int> > noeud_arbre::get_equivalence() {
  return equivalence;
}

void noeud_arbre::display_progress(int &ini,  const int &total, int &carrier, const int &sp) {
  int true_percent;
  double percent = (ini * 1.) / (total * 1.);
  percent *= 100;
  percent = floor(percent);
  true_percent = (int) percent;
  if (true_percent % sp == 0 && carrier != true_percent) {
    py::print(true_percent, "%", "sep"_a="");
    carrier = true_percent;
  }
}

void noeud_arbre::vite(const int &sp) {
  int ct = 0, k, t=0;
  // equivalence carrier ----
  bool eq_true = false;
  std::vector<int> eq(2, 0);
  //
  std::cout << "Commencer: la abre a merde\n";
  // From distance matrix to upper triangular array
  double* triangular_matrix = new double[(leaves * (leaves - 1)) / 2];
  for (int i = 0; i < distance.size(); i++) {
    triangular_matrix[i] = distance[i];
  }
  // hclust
  int* labels = new int[leaves];
  int* merge = new int[2 * (leaves - 1)];
  double* height = new double[leaves - 1];
  hclust_fast(
    leaves,
    triangular_matrix,
    linkage, // linkage method
    merge,
    height
  );
  // Get NEC >= 1
  std::vector<int> nec_more_one = NEC_one(NEC, n);
  // Create merge list
  std::vector<node> merge_list(nodes);
  for (int i = 0; i < nodes; i++) {
    merge_list[i].merge = i;
    merge_list[i].members.push_back(i);
  }
  // The loop starts!! - look over link dendrogram - tie 0
  for (int i = 0; i < nec_more_one.size(); i++) {
    display_progress(i, leaves - 1, t, sp);
    k = K[nec_more_one[i]];
    double h = height_feature[nec_more_one[i]];
    cutree_k(
      leaves,
      merge,
      k,
      labels
    );
    // unique labels
    std::vector<int> unique_labels(labels, labels + leaves);
    unique(unique_labels);
    // Compute Dc
    std::vector<double> Dcs = link_communitiy_Dc(
      labels, leaves,
      source, target, undirected
    );
    // Valid DCs
    std::vector<int> valid_DCs = DC_valid(Dcs);
    // Clone merge_list
    std::vector<node> merge_list_clone = merge_list;
    // The loop across valid LCs - tier 1
    for (int j = 0; j < valid_DCs.size(); j++) {
      std::vector<int> node_src_1;
      std::vector<int> node_tgt_1;
      // Get source and target nodes
      for (int kk = 0; kk < leaves; kk++) {
        if (labels[kk] == unique_labels[valid_DCs[j]]) {
          if (!undirected) {
            node_src_1.push_back(source[kk]);
            node_tgt_1.push_back(target[kk]);
          } else {
            node_src_1.push_back(source[kk]);
            node_tgt_1.push_back(target[kk]);
            node_src_1.push_back(target[kk]);
            node_tgt_1.push_back(source[kk]);
          }
        }
      }
      unique(node_src_1);
      unique(node_tgt_1);
      std::vector<int> inter_1 = intersect(node_src_1, node_tgt_1);
      // If LC is a non-trivial community
      if (inter_1.size() > 1) {
        std::vector<int> compatible_LCs;
        for (int kk = 0; kk < merge_list_clone.size(); kk++) {
          std::vector<int> inter_2 = intersect(
            inter_1, merge_list_clone[kk].members
          );
          if (inter_2.size() >= 1) compatible_LCs.push_back(kk);
        }
        // Only 2 compatible LCs
        if (compatible_LCs.size() == 2) {
          // eq true
          eq_true = true;
          //
          node_hierarchy[ct][0] = merge_list_clone[compatible_LCs[0]].merge;
          node_hierarchy[ct][1] = merge_list_clone[compatible_LCs[1]].merge;
          node_hierarchy[ct][2] = h;
          node_hierarchy[ct][3] = merge_list_clone[compatible_LCs[0]].members.size() + merge_list_clone[compatible_LCs[1]].members.size();
          // Merge new members
          std::vector<int> new_members = merge_list_clone[compatible_LCs[0]].members;
          new_members.insert(
            new_members.end(),
            merge_list_clone[compatible_LCs[1]].members.begin(),
            merge_list_clone[compatible_LCs[1]].members.end()
          );
          node new_node = {
            ct + nodes,
            new_members
          };
          // Erase elements
          for (int kk = 0; kk < 2; kk++) {
            merge_list_clone.erase(
              merge_list_clone.begin() +
              compatible_LCs[kk] - kk
            );
          }
          // Merge new node ----
          merge_list_clone.push_back(new_node);
          // eq
          eq[0] = k;
          eq[1] = nodes - ct - 1;
          equivalence.push_back(eq);
          //
          ct++;
        }
        // If there more than 2 compatible LCs
        if (compatible_LCs.size() > 2) {
          // eq true
          eq_true = true;
          //
          // Clone merge_list_clone
          std::vector<node> merge_list_clone_2;
          for (int kk = 0; kk < compatible_LCs.size(); kk++) {
            merge_list_clone_2.push_back(
              merge_list_clone[compatible_LCs[kk]]
            );
          }
          for (int kk = 0; kk < compatible_LCs.size(); kk++) {
            merge_list_clone.erase(
              merge_list_clone.begin() +
              compatible_LCs[kk] - kk
            );
          }
          while (merge_list_clone_2.size() >= 2) {
            // merge first two nodes in the list
            node_hierarchy[ct][0] = merge_list_clone_2[0].merge;
            node_hierarchy[ct][1] = merge_list_clone_2[1].merge;
            node_hierarchy[ct][2] = h;
            node_hierarchy[ct][3] = merge_list_clone_2[0].members.size() + merge_list_clone_2[1].members.size();
            if (merge_list_clone_2.size() > 2) {
              // Merge new members
              std::vector<int> new_members = merge_list_clone_2[0].members;
              new_members.insert(
                new_members.end(),
                merge_list_clone_2[1].members.begin(),
                merge_list_clone_2[1].members.end()
              );
              unique(new_members);
              node new_node = {
                ct + nodes,
                new_members
              };
              merge_list_clone_2.erase(
                merge_list_clone_2.begin(),
                merge_list_clone_2.begin() + 2
              );
              merge_list_clone_2.push_back(new_node);
            } else if (merge_list_clone_2.size() == 2) {
              // Merge new members
              std::vector<int> new_members = merge_list_clone_2[0].members;
              new_members.insert(
                new_members.end(),
                merge_list_clone_2[1].members.begin(),
                merge_list_clone_2[1].members.end()
              );
              unique(new_members);
              node new_node = {
                ct + nodes,
                new_members
              };
              merge_list_clone_2.erase(
                merge_list_clone_2.begin(),
                merge_list_clone_2.begin() + 2
              );
              merge_list_clone.push_back(new_node);
            }
            // eq
            eq[0] = k;
            eq[1] = nodes - ct - 1;
            equivalence.push_back(eq);
            //
            ct++;
          }
        }
        merge_list = merge_list_clone;
      }
    }
    if (!eq_true) {
      eq[0] = k;
      eq[1] = nodes - ct - 1;
      equivalence.push_back(eq);
    } else {
      eq_true = false;
    }
  }
  if (merge_list.size() > 1) {
    std::cout << "Adding exceptional nodes." << std::endl;
    for (int kk = 0; kk < merge_list.size() - 1; kk++) {
      node_hierarchy[ct][0] = merge_list.back().merge;
      node_hierarchy[ct][1] = merge_list[kk].merge;
      node_hierarchy[ct][2] = node_hierarchy[ct - 1][2];
      node_hierarchy[ct][3] = merge_list.back().members.size() + merge_list[kk].members.size();
      merge_list.back().merge = ct + nodes;
      merge_list.back().members.push_back(merge_list[kk].members[0]);
      // eq
      eq[0] = k;
      eq[1] = nodes - ct - 1;
      equivalence.push_back(eq);
      //
      ct++;
    }
  }
  equivalence[equivalence.size() - 1][1] = 1;
  delete[] labels;
  delete[] merge;
  delete[] height;
  delete[] triangular_matrix;
  std::cout << "Voila, bon ami\n";
}

PYBIND11_MODULE(la_arbre_a_merde, m) {
    py::class_<noeud_arbre>(m, "noeud_arbre")
        .def(
          py::init<
            std::vector<double>,
            std::vector<int>,
            std::vector<int>,
            std::vector<int>,
            std::vector<double>,
            std::vector<int>,
            const int,
            const int,
            const int,
            const int,
            const int,
            const bool
          >()
        )
        .def("vite", &noeud_arbre::vite)
        .def("get_node_hierarchy", &noeud_arbre::get_node_hierarchy)
        .def("get_equivalence", &noeud_arbre::get_equivalence)
				.def("display_progress", &noeud_arbre::display_progress);
}