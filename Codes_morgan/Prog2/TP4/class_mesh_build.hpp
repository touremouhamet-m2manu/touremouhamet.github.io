// // Dans le fichier class_mesh.hpp, ajouter après les includes:
// #include "class_face.hpp"  // Déjà présent
// #include "row_element.hpp"  // Pour connectivity_data (à adapter si nécessaire)

// // ... puis dans la classe mesh, modifier la méthode build_edges:

// template <class T>
// void mesh<T>::build_edges(std::size_t n_vertices_)
// {
//     int n_elements = this->length();
    
//     // CORRECTION: Utiliser connectivity_data au lieu de std::size_t
//     list<linked_list<connectivity_data>> hashv(n_vertices_);

//     std::vector<std::size_t> adj_vertices(n_vertices_, 0);
//     std::size_t n_edges = 0;

//     // CORRECTION: Pointeurs adaptés pour connectivity_data
//     linked_list<connectivity_data>* pc_cell = nullptr;
//     linked_list<connectivity_data>* pc_cell_prev = nullptr;

//     // pointers to vertices
//     vertex* scan_min = nullptr;
//     vertex* scan_max = nullptr;

//     size_t ismin;
//     size_t ismax;

//     size_t edges_counter = 0;

//     // CORRECTION: Nouvelle boucle avec index d'élément courant
//     for (mesh<T>* scanner = this; scanner; scanner = (mesh<T>*)scanner->p_next())
//     {
//         auto current_element_index = scanner->item().index();
//         // triangle* p_current_element = &scanner->item(); // si besoin

//         for (auto kk = 0; kk < 3; kk++)
//         {
//             size_t next = (kk + 1) % 3;

//             if (scanner->item().vertex(kk)->index() <= scanner->item().vertex(next)->index())
//             {
//                 scan_min = scanner->item().vertex(kk);
//                 scan_max = scanner->item().vertex(next);
//                 ismin = scan_min->index();
//                 ismax = scan_max->index();
//             }
//             else
//             {
//                 scan_min = scanner->item().vertex(next);
//                 scan_max = scanner->item().vertex(kk);
//                 ismin = scan_min->index();
//                 ismax = scan_max->index();
//             }

//             if (!hashv.item(ismin))
//             {
//                 // CORRECTION: Créer avec connectivity_data
//                 hashv.item(ismin) = new linked_list<connectivity_data>(ismin);
//             }

//             // CORRECTION: Parcourir la liste chaînée de connectivity_data
//             pc_cell_prev = hashv.item(ismin);
//             pc_cell = hashv.item(ismin)->p_next();

//             while (pc_cell && (pc_cell->item().vertex_index() <= ismax))
//             {
//                 pc_cell_prev = pc_cell;
//                 pc_cell = pc_cell_prev->p_next();
//             }

//             if (pc_cell_prev->item().vertex_index() < ismax)
//             {
//                 // CORRECTION: Cas 1 - Nouvelle arête
//                 // Créer un objet connectivity_data avec les données
//                 connectivity_data tmp(ismax, edges_counter, current_element_index);
//                 pc_cell_prev->insert_next_item(tmp);
                
//                 // Ajouter cette arête comme face de l'élément avec l'index local kk
//                 scanner->item().set_face_index(kk, edges_counter);
                
//                 ++adj_vertices[ismin];
//                 ++n_edges;
                
//                 // CORRECTION: Utiliser le nouveau constructeur de edge
//                 edges_.push_back(
//                     edge(*scan_min, *scan_max, edges_counter++, current_element_index, -1));
//             }
//             else
//             {
//                 // CORRECTION: Cas 2 - Arête déjà existante
//                 // Mettre à jour l'index du deuxième triangle voisin
//                 edges_.at(pc_cell_prev->item().edge_index()).neighbor(1) = current_element_index;
                
//                 // Ajouter cette arête comme face de l'élément avec l'index local kk
//                 scanner->item().set_face_index(kk, pc_cell_prev->item().edge_index());
//             }
//         }
//     }

//     std::ofstream os;
//     os.open("segment.txt");

//     for (auto& this_edge : edges_)
//     {
//         os << this_edge[0].location() << std::endl;
//         os << this_edge[1].location() << std::endl;
//         os << std::endl;
//     }
//     os.close();

//     std::cout << "n_edges= " << n_edges << std::endl;
//     std::cout << " -- exit build_edges -- " << std::endl << std::endl;
// }