#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include<ctime> // time
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

std::vector<std::vector<int> > create_id_matrix(
	std::vector<std::vector<bool> >& matrix,
	const int& M, const int& N
) {
	std::vector<std::vector<int> > id_matrix(N, std::vector<int>(M, 0));
	int id = 1;
	for (int i=0; i < M; i++){
		for (int j=0; j < N; j++){
			if (matrix[i][j] > 0){
				id_matrix[i][j] = id;
				id++;
			}
		}
	}
	return id_matrix;
}

class simquest {
  private:
    std::vector<double> linksim_matrix;
    std::vector<std::vector<double> > source_matrix;
    std::vector<std::vector<double> > target_matrix;
  public:
    simquest(
      std::vector<std::vector<bool> > BA,
      std::vector<std::vector<double> > AKI,
      std::vector<std::vector<double> > AIK,
      const int N,
      const int leaves,
      const int topology,
      const int index
    );
    ~simquest(){};
    std::vector<double> calculate_linksim_matrix(
      std::vector<std::vector<bool> >& bmatrix, const int& N, const int& leaves
    );
    double similarity_index(
      std::vector<double> &u, std::vector<double> &v, int &ii, int &jj, const int &index
    );
    std::vector<std::vector<double> > calculate_nodesim_matrix(
      std::vector<std::vector<double> >& matrix, const int& N, const int& index
    );
		std::vector<double> get_linksim_matrix();
		std::vector<std::vector<double> > get_source_matrix();
		std::vector<std::vector<double> > get_target_matrix();
		double jacp(std::vector<double> &u, std::vector<double> &v, int &ii, int &jj);
		double jacw(std::vector<double> &u, std::vector<double> &v, int &ii, int &jj);
		double tanimoto_coefficient(std::vector<double> &u, std::vector<double> &v, int &ii, int &jj);
		double cosine_similarity(std::vector<double> &u, std::vector<double> &v, int &ii, int &jj);
		double S1_2(std::vector<double> &u, std::vector<double> &v, int &ii, int&jj);
		double H2(std::vector<double> &u, std::vector<double> &v, int &ii, int&jj);
		double bin_similarity(std::vector<double> &u, std::vector<double> &v, int &ii, int &jj);
};

simquest::simquest(
	std::vector<std::vector<bool> > BA,
  std::vector<std::vector<double> > AKI,
  std::vector<std::vector<double> > AIK,
	const int N,
	const int leaves,
  const int topology,
  const int index
){
	// MIX topology
	if (topology == 0) {
		source_matrix = calculate_nodesim_matrix(AIK, N, index);
		target_matrix = calculate_nodesim_matrix(AKI, N, index);
	}
	// SOURCE topology
	else if (topology == 1) {
		source_matrix = calculate_nodesim_matrix(AIK, N, index);
		target_matrix = source_matrix;
	}
	// TARGET topology
	else if (topology == 2) {
		source_matrix = calculate_nodesim_matrix(AKI, N, index);
		target_matrix = source_matrix;
	}
	linksim_matrix = calculate_linksim_matrix(BA, N, leaves);
}

std::vector<std::vector<double> > simquest::calculate_nodesim_matrix(
	 std::vector<std::vector<double> >& matrix, const int& N, const int& index
) {
	std::vector<std::vector<double> > node_sim_matrix(N, std::vector<double>(N, 0.));
	for (int i=0; i < N; i++) {
		for (int j=i; j < N; j++) {
			if (i == j) continue;
			node_sim_matrix[i][j] = similarity_index(matrix[i], matrix[j], i, j, index);
			node_sim_matrix[j][i] = node_sim_matrix[i][j];
		}
	}
	return node_sim_matrix;
}

std::vector<double> simquest::calculate_linksim_matrix(
	std::vector<std::vector<bool> >& matrix, const int& N, const int& leaves
) {
	int t;
	std::vector<std::vector<int> > id_matrix = create_id_matrix(matrix, N, N);

	t = (int) ((leaves - 1.) * leaves / 2.);
	std::vector<double> link_similarity_matrix(t, 0.);
	int col_id, row_id;
	for (int i =0; i < N; i++) {
		for (int j=0; j < N; j++) {
			row_id = id_matrix[i][j];
			if (row_id == 0) continue;
			for (int k=j; k < N; k++) {
				col_id = id_matrix[i][k];
				if (k == j || col_id == 0) continue;
				t = leaves * (row_id - 1) + col_id - 1 - 2 * (row_id -1) - 1;
				t -= (int) ((row_id - 1.) * (row_id - 2.) / 2);
				link_similarity_matrix[t] = target_matrix[j][k];
			}
			for (int k=i; k < N; k++) {
				col_id = id_matrix[k][j];
				if (k == i || col_id == 0) continue;
				t = leaves * (row_id - 1) + col_id - 1 - 2 * (row_id -1) - 1;
				t -= (int) ((row_id - 1.) * (row_id - 2.) / 2);
				link_similarity_matrix[t] = source_matrix[i][k];
			}
		}
	}
	return link_similarity_matrix;
}

double simquest::tanimoto_coefficient(
	std::vector<double> &u, std::vector<double> &v , int &ii, int &jj
) {
	int N = u.size();
	double uv=0., uu=0., vv=0.;
	for (int i=0; i < N; i++) {
		if (i == ii || i == jj) continue;
		uv += u[i] * v[i];
		uu += u[i] * u[i];
		vv += v[i] * v[i];
	}
	uv += u[jj] * v[ii];
	uu += u[jj] * u[jj];
	vv += v[ii] * v[ii];

	uv += u[ii] * v[jj];
	uu += u[ii] * u[ii];
	vv += v[jj] * v[jj];
	return uv / (uu + vv - uv);
}

double simquest::cosine_similarity(
	std::vector<double> &u, std::vector<double> &v, int &ii, int &jj
) {
	int N = u.size();
	double uv=0., uu=0., vv=0.;
	for (int i=0; i < N; i++) {
		if (i == ii || i == jj) continue;
		uv += u[i] * v[i];
		uu += u[i] * u[i];
		vv += v[i] * v[i];
	}
	uv += u[jj] * v[ii];
	uu += u[jj] * u[jj];
	vv += v[ii] * v[ii];

	uv += u[ii] * v[jj];
	uu += u[ii] * u[ii];
	vv += v[jj] * v[jj];
	return uv / (sqrt(uu * vv));
}

double simquest::S1_2(
	std::vector<double> &u, std::vector<double> &v, int &ii, int &jj
) {
	int N = u.size();
	double JACP = 0.;
	double p = 0, pu = 0, pv = 0;
	for (int j=0; j < N; j++){
		pu += u[j];
		pv += v[j];
	}
	for (int i=0; i < N; i++){
		// D1/2
		if (i == ii | i == jj) continue;
		p += sqrt(((u[i]) / pu) * ((v[i]) / pv));
	}
	p += sqrt((u[jj]) / pu * ((v[ii]) / pv));
	p += sqrt((u[ii]) / pu * ((v[jj]) / pv));
	// S1/2
  if (p > 0) {
    p = - 2 * log(p);
    JACP = 1 / (1 + p);
  }
  else
    JACP = 0;
	return JACP;
}

double simquest::H2(
	std::vector<double> &u, std::vector<double> &v, int &ii, int &jj
) {
	int N = u.size();
	double p = 0, pu = 0, pv = 0;
	for (int j=0; j < N; j++){
		pu += u[j];
		pv += v[j];
	}
	if (pu > 0 && pv > 0) {
		for (int i=0; i < N; i++){
			if (i == ii | i == jj) continue;
			p += pow(sqrt(u[i] / pu)  - sqrt(v[i] / pv), 2.);
		}
		if (ii < N && jj < N) {
			p += pow(sqrt(u[jj] / pu) - sqrt(v[ii] / pv), 2.);
			p += pow(sqrt(u[ii] / pu) - sqrt(v[jj] / pv), 2.);
		}

		return 1. - (0.5 * p);
	}
	else {
		return 0.;
	}
}

double simquest::bin_similarity(
	std::vector<double> &u, std::vector<double> &v, int &ii, int &jj
) {
	int N = u.size();
	double uv=0., uu=0., vv=0.;
	for (int i=0; i < N; i++) {
		if (i == ii | i == jj) continue;
		if ((u[i] > 0 && v[i] > 0) || (u[i] == 0 && v[i] == 0)) uv++;
		if (u[i] > 0) uu++;
		if (v[i] > 0) vv++;
	}

	if ((u[jj] > 0 && v[ii] > 0) || (u[jj] == 0 && v[ii] == 0)) uv++;
	if (u[jj] > 0) uu++;
	if (v[ii] > 0) vv++;

	if ((u[ii] > 0 && v[jj] > 0) || (u[ii] == 0 && v[jj] == 0)) uv++;
	if (u[ii] > 0) uu++;
	if (v[jj] > 0) vv++;

	uv /= N;
	uu /= N;
	vv /= N;

	return uv - 1 + uu + vv - (2 * uu * vv);
}

double simquest::jacp(
	std::vector<double> &u, std::vector<double> &v, int &ii, int &jj)
{
	int N = u.size();
	double JACP = 0;
	double p;
	for (int i=0; i < N; i++){
		if ((u[i] > 0 && v[i] > 0) && (i != ii || i != jj)){
			p = 0;
			for (int j=0; j < N; j++) {
				if (j == ii || j == jj) continue;
				p += std::max(u[j]/u[i], v[j]/v[i]);
			}
			p += std::max(u[jj]/u[i], v[ii]/v[i]);
			p += std::max(u[ii]/u[i], v[jj]/v[i]);
			if (p != 0)
				JACP += 1 / p;
			else
				std::cout << "Vectors in jaccardp are both zero\n";
		}
	}

	if (u[jj] > 0 && v[ii] > 0) {
		p = 0;
		for (int j=0; j < N; j++) {
			if (j == ii || j == jj) continue;
			p += std::max(u[j]/u[jj], v[j]/v[ii]);
		}
		p += std::max(u[ii]/u[jj], v[jj]/v[ii]);
		p += 1;
		if (p != 0)
			JACP += 1 / p;
		else
			std::cout << "Vectors in jaccardp are both zero\n";
	}

	if (u[ii] > 0 && v[jj] > 0) {
		p = 0;
		for (int j=0; j < N; j++) {
			if (j == ii || j == jj) continue;
			p += std::max(u[j]/u[jj], v[j]/v[ii]);
		}
		p += std::max(u[jj]/u[ii], v[ii]/v[jj]);
		p += 1;
		if (p != 0)
			JACP += 1 / p;
		else
			std::cout << "Vectors in jaccardp are both zero\n";
	}
	return JACP;
}

double simquest::jacw(
	std::vector<double> &u, std::vector<double> &v, int &ii, int &jj
) {
	int N = u.size();
	double mi =0., ma=0.;
	for (int i=0; i < N; i++) {
		if (i == ii || i == jj) continue;
		mi += std::min(u[i], v[i]);
		ma += std::max(u[i], v[i]);
	}
	mi += std::min(u[jj], v[ii]);
	ma += std::max(u[jj], v[ii]);

	mi += std::min(u[ii], v[jj]);
	ma += std::max(u[ii], v[jj]);
	return mi / ma;
}

double simquest::similarity_index(std::vector<double> &u, std::vector<double> &v, int &ii, int &jj, const int &index) {
	// Jaccard probability index
  if (index == 0) {
    return jacp(u, v, ii, jj);
  }
	// Tanimoto coefficient
  else if (index == 1) {
		return tanimoto_coefficient(u, v, ii, jj);
	}
	// Cosine similarity
  else if (index == 2) {
		return cosine_similarity(u, v, ii, jj);
	}
	else if (index == 3) {
		return bin_similarity(u, v, ii, jj);
	}
	else if (index == 4) {
		return S1_2(u, v, ii, jj);
	}
	else if (index == 5) {
		return jacw(u, v, ii, jj);
	}
	else if (index == 6) {
		return H2(u, v, ii, jj);
	}
  else {
    std::range_error("Similarity index must be a integer from 0 to 4\n");
  }
}

std::vector<double> simquest::get_linksim_matrix() {
	return linksim_matrix;
}

std::vector<std::vector<double> > simquest::get_source_matrix() {
	return source_matrix;
}

std::vector<std::vector<double> > simquest::get_target_matrix() {
	return target_matrix;
}

PYBIND11_MODULE(simquest, m) {
    py::class_<simquest>(m, "simquest")
        .def(
          py::init<
           std::vector<std::vector<bool> >,
						std::vector<std::vector<double> >,
						std::vector<std::vector<double> >,
						const int,
						const int,
						const int,
						const int
          >()
        )
        .def("get_linksim_matrix", &simquest::get_linksim_matrix)
        .def("get_source_matrix", &simquest::get_source_matrix)
				.def("get_target_matrix", &simquest::get_target_matrix);
}