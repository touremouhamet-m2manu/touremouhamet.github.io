#include<iostream>
#include"class_element.hpp"
#include"class_node.hpp"
#include"class_linked_list.hpp"

typedef std::array<int, 3> itriplet;

template<class T>
class mesh : public linked_list
{   
    private:
    // mesh(T& e) { item = e; } 
    public:
    mesh(T& e):linked_list<T>(e){} // constructor

    int indexing ();

};

typedef mesh<triangle> triangulation;

template<class T>
int mesh<T>:: indexing ( ) 
{
    for (mesh<T>* p_scan=this ; p_scan; p_scan=(mesh<T>*)p_scan−>p_next)
    {
        p_scan−>item.reset_indices();
        // p_scan->item().print_vertices();
    }

    int count=1;
    for (mesh<T>* p_scan =this; p_scan; p_scan=(mesh<T>*)p_scan−>p_next)
    {
        p_scan−>item. indexing(count );
        p_scan->item().print_vertices();
    }
return count;
} // indexing the nodes in the mesh

// POPULATE VERTICES AND TRIANGLES FROM FILE's data
bool mesh_reader(std::string const fname,
                 std::vector<vertex*>& vertices,
                 std::vector<triangle>& triangles)

if (!mesh_file)



// skip 7 first lines
// for (int irow=1; irow<=7; ++irow)
// getline(mesh_file, line_);

while (marker.compare("vertices"))
{
    mesh_file >> marker;
    assert(!mesh_file.eof());
}

int n_nodes;
mesh_file >> n_nodes;



coords node;
std::vector<point2d> points;

for (auto in =0; in < n_nodes; in++)
{
    mesh_files >> node.x >> node.y >> node.index
}

// **********************************************************************
template<class T>
void mesh<T>::build_edges(std::vector<edge>& edges, std::size_t n_vertices)
{
    int n_elements = this->length();
    std::cout << "n_vertices=" << n_vertices << std::endl;

    // data structure to store vectices-to-edges data
    list<linked_list
}



// looping on elements->next
