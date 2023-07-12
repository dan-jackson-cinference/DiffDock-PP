def get_four_letters_pdb_identifier(pdb_name: str):
    return pdb_name.split("/")[-1].split(".")[0]


def write_pdb(item, graph, part, path):
    lines = to_pdb_lines(item, graph, part)
    with open(path, "w") as file:
        file.writelines(lines)


def to_pdb_lines(visualization_values, graph, part):
    assert part in ("ligand", "receptor", "both"), "Part should be ligand or receptor"
    parts = ["ligand", "receptor"] if part == "both" else [part]

    lines = []
    for part in parts:
        this_vis_values = visualization_values[part]
        this_vis_values = {
            k: v.strip() if type(v) is str else v for k, v in this_vis_values.items()
        }
        for i, resname in enumerate(this_vis_values["resname"]):
            xyz = graph[part].pos[i]

            line = f'ATOM  {i + 1:>5} {this_vis_values["atom_name"][i]:>4} '
            line = line + f'{resname} {this_vis_values["chain"][i]}{this_vis_values["residue"][i]:>4}    '.replace(
                "<Chain id=", ""
            ).replace(
                ">", ""
            )
            line = line + f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}"
            line = line + "  1.00  0.00          "
            line = line + f'{this_vis_values["element"][i]:>2} 0\n'
            lines.append(line)

    return lines
