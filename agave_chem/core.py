from rdkit import Chem
from rdkit.Chem import AllChem
from rdchiral import main as rdc
import json
from ast import literal_eval
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SMIRKS_PATTERNS_FILE = BASE_DIR / 'datafiles' / 'smirks_patterns.json'

class AgaveChemMapper():
    """
    AgaveChem reaction classification and atom-mapping
    """
    def __init__(self):
        smirks_patterns = []
        with open(SMIRKS_PATTERNS_FILE, 'rb') as f:
            lines = f.readlines()
            for line in lines:
                smirks_patterns.append(literal_eval(line.decode('utf-8')))

        self.smirks_name_dictionary = {ele['smirks']:{'name': ele['name'], 'superclass': ele['superclass_id']} for ele in smirks_patterns}
        self.smirks_patterns = self._initialize_template_data(smirks_patterns)
        self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

    def _expand_first_bracketed_list(self, input_string):
        """
        Finds the first bracketed section containing a field with commas
        (e.g., [C;H1,H2;+0:6]) and expands that specific field.
    
        It iterates through brackets until it finds one needing expansion, expands
        only the *first* comma-separated field within that bracket, and returns
        all possible strings resulting from that single expansion step.
    
        Args:
            input_string: The string to process.
    
        Returns:
            A list of strings, where each string represents one expansion
            of the first found comma-separated field within a bracket.
            If no such field is found in any bracket, returns a list containing
            just the original string.
        """
    
        bracket_matches = []
        open_bracket_count, close_bracket_count = 0, 0
        start_index, end_index = 0, 0
        for i, char in enumerate(input_string):
            if char == '[':
                if open_bracket_count == 0:
                    start_index = i
                open_bracket_count+=1
                
            if char == ']':
                end_index = i+1
                close_bracket_count+=1
    
            if open_bracket_count==close_bracket_count and open_bracket_count!=0:
                bracket_matches.append([input_string[start_index:end_index], start_index, end_index])
                open_bracket_count = 0
                close_bracket_count = 0
    
        for original_bracket_with_brackets, start_index, end_index in bracket_matches:
            original_bracket_content = original_bracket_with_brackets[1:-1]
            has_map = False
            if len(original_bracket_content.split(':')) > 1:
                sub_str = original_bracket_content.split(':')[-1]
                if sub_str.isdigit() or (sub_str.startswith('-') and sub_str[1:].isdigit()):
                    has_map = True
                    map_num = original_bracket_content.split(':')[-1]
            if has_map:
                original_bracket_content_without_map = ':'.join(original_bracket_content.split(':')[:-1])
            else:
                original_bracket_content_without_map = original_bracket_content
    
            fields = original_bracket_content_without_map.split(';')
    
            field_to_expand_index = -1
            alternatives = []
            for i, field in enumerate(fields):
                if ',' in field:
                    field_to_expand_index = i
                    alternatives = field.split(',') 
                    break 
    
            if field_to_expand_index != -1:
                results = []
                for alt in alternatives:
                    new_fields = fields[:field_to_expand_index] + [alt] + fields[field_to_expand_index + 1:]
                    new_bracket_content = ';'.join(new_fields)
                    if has_map:
                        new_bracket_content += ':' + map_num
                    new_string = (
                        input_string[:start_index] +      
                        '[' + new_bracket_content + ']' +   
                        input_string[end_index:]            
                    )
                    results.append(new_string)
    
                return results
    
        return [input_string]

    def _expand_all_recursively(self, input_string):
        """
        Recursively applies _expand_first_bracketed_list to a SMIRKS string until
        no more expansions based on comma-separated fields within brackets are
        possible. It collects all final, fully expanded combinations.
    
        Args:
            input_string: The SMIRKS string to start the expansion from.
    
        Returns:
            A list of all fully expanded SMIRKS strings.
        """
    
        expanded_list = self._expand_first_bracketed_list(input_string)
    
        if len(expanded_list) == 1 and expanded_list[0] == input_string:
            return expanded_list
    
        else:
            all_final_strings = []
            for next_string in expanded_list:
                all_final_strings.extend(self._expand_all_recursively(next_string))
            return all_final_strings

    def _initialize_template_data(self, named_reactions):
        """
        Initialize reaction template data by processing SMIRKS patterns from named reactions.
        
        This function takes a list of named reactions, reverses their SMIRKS patterns,
        expands them recursively, and creates RDChiral reaction objects for further processing.
        
        Args:
            named_reactions (list): List of dictionaries containing named reaction information.
                                   Each dictionary should have a 'smirks' key.
        
        Returns:
            list: List of lists, where each inner list contains:
                  [0] - List of product SMARTS molecules
                  [1] - List of reactant SMARTS molecules
                  [2] - RDChiral reaction object
                  [3] - Parent SMIRKS 
                  [4] - Child SMIRKS
        """
        all_smirks = {}
        for reaction in named_reactions:
            smirks_list = []
            smirks = reaction['smirks'].split('>>')[1] + '>>' + reaction['smirks'].split('>>')[0]
            
            
            if len(self._expand_all_recursively(smirks)) < 100:
                smirks_list.extend(self._expand_all_recursively(smirks))

            all_smirks[smirks] = smirks_list
        
        rdc_info = []
        for k,v in all_smirks.items():
            for smirk in v:
                try:
                    products_smarts = [Chem.MolFromSmarts(ele) for ele in smirk.split('>>')[0].split('.')]
                    reactants_smarts = [Chem.MolFromSmarts(ele) for ele in smirk.split('>>')[1].split('.')]
                    
                    rdc_rxn = rdc.rdchiralReaction(smirk)
                    
                    rdc_info.append([
                        products_smarts, 
                        reactants_smarts,
                        rdc_rxn, 
                        k,
                        smirk
                    ])
                except:
                    pass
        
        return rdc_info

    def _transfer_mapping(self, mapped_substructure_smarts, full_molecule_smiles):
        """
        Transfers atom map numbers from a mapped SMARTS substructure
        to a full molecule SMILES, leaving atoms corresponding to '*' unmapped.
    
        Args:
            mapped_substructure_smarts (str): SMARTS string of the substructure
                                               with atom map numbers. Wildcards (*)
                                               are expected for connection points
                                               and should not have map numbers.
            full_molecule_smiles (str): SMILES string of the complete, unmapped molecule.
    
        Returns:
            str: The SMILES string of the full molecule with map numbers transferred
                 from the substructure match, or None if an error occurs (e.g.,
                 parsing failed, substructure not found).
        """
        pattern = Chem.MolFromSmarts(mapped_substructure_smarts)
        mol = Chem.MolFromSmiles(full_molecule_smiles)
        match_indices = mol.GetSubstructMatches(pattern)
    
        symmetry_class = {k:v for k,v in enumerate(list(Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False)))}
        
        symmetric=True
        for match_1 in match_indices:
            for match_2 in match_indices:
                for ele1, ele2 in zip(match_1, match_2):
                    if symmetry_class[ele1] != symmetry_class[ele2]:
                        symmetric = False
    
        if not match_indices:
            return None
    
        if len(match_indices) != 1 and not symmetric:
            return None
    
        match_indices = match_indices[0]
    
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() != 0:
                atom.SetAtomMapNum(0)
    
        for pattern_atom in pattern.GetAtoms():
            map_num = pattern_atom.GetAtomMapNum()
    
            if map_num > 0:
                pattern_idx = pattern_atom.GetIdx()
                mol_idx = match_indices[pattern_idx]
                mol_atom = mol.GetAtomWithIdx(mol_idx)
                mol_atom.SetAtomMapNum(map_num)
    
        mapped_smiles_output = Chem.MolToSmiles(mol)
        return mapped_smiles_output

    def _canonicalize_smiles(self, smiles, isomeric=True, remove_mapping=True, canonicalize_tautomer=True):
        """
        Canonicalizes a SMILES string, handling multiple fragments, atom mapping, and tautomers.

        This function takes a SMILES string, which may represent multiple disconnected
        fragments separated by '.', canonicalizes each fragment individually,
        optionally removes atom map numbers, optionally canonicalizes tautomers,
        and returns a single SMILES string with the canonical fragments sorted
        alphabetically and joined by '.'.

        Args:
            smiles (str): The input SMILES string.
            isomeric (bool, optional): If True, includes stereochemistry information
                in the output canonical SMILES. Defaults to True.
            remove_mapping (bool, optional): If True, removes atom map numbers from
                the molecules before canonicalization. Defaults to True.
            canonicalize_tautomer (bool, optional): If True, canonicalizes the
                tautomeric form of each fragment using self.tautomer_enumerator.
                Defaults to True.

        Returns:
            str: The canonicalized SMILES string with fragments sorted alphabetically.
                 Returns the original input SMILES string if any processing error occurs.
        """
        try:
            x = smiles.split('.')
            x = sorted(x)
            frags = []
            for i in x:
                m = Chem.MolFromSmiles(i)
                if remove_mapping:
                    [a.SetAtomMapNum(0) for a in m.GetAtoms()]
                if canonicalize_tautomer:
                    m = self.tautomer_enumerator.Canonicalize(m)
                canonical_smiles_string = str(Chem.MolToSmiles(m, canonical=True, isomericSmiles=isomeric))
                frags.append(canonical_smiles_string)
            canonical_smiles_string = '.'.join(i for i in sorted(frags) if i != '')
            return(canonical_smiles_string)
        except:
            return smiles
    
    def _canonicalize_reaction_smiles(self, rxn_smiles, isomeric=True):
        """
        Canonicalizes the SMILES representation of each molecule within a reaction SMILES.

        This function takes a reaction SMILES string (reactants>[agents]>products),
        splits it into its components (reactants, agents, products), canonicalizes
        the SMILES string for each molecule within each component using the
        _canonicalize_smiles method, sorts the molecules alphabetically within
        each component, and reassembles the reaction SMILES string.

        Args:
            rxn_smiles (str): The input reaction SMILES string (e.g., "CC.O>>CCO").
            isomeric (bool, optional): If True, includes stereochemistry information
                when canonicalizing individual molecule SMILES. Passed to
                _canonicalize_smiles. Defaults to True.

        Returns:
            str: The canonicalized reaction SMILES string, with molecules in each
                 part (reactants, agents, products) individually canonicalized
                 and sorted alphabetically. Returns the original input reaction
                 SMILES if any processing error occurs.
        """
        try:
            split_roles = rxn_smiles.split('>>')
            reaction_list = []
            for x in split_roles:
                role_list = []
                if x != '':
                    y = x.split('.')
                    for z in y:
                        canonical_smiles = self._canonicalize_smiles(z, isomeric)
                        role_list.append(canonical_smiles)
        
                    role_list = sorted(role_list)
                    role_list = [ele for ele in role_list if ele != '']
                    reaction_list.append(role_list)
        
            canonical_rxn = ['.'.join(role_list) for role_list in reaction_list]
            canonical_rxn = '>>'.join(canonical_rxn)
            return(canonical_rxn)
        except:
            return rxn_smiles

    def _canonicalize_atom_mapping(self, reaction_smiles):
        """
        Canonicalizes atom mapping numbers for a reaction SMILES.
        
        This function identifies corresponding atoms in the reactants and products
        side of a reaction SMILES, then modifies the atom-mapping numbers to canonicalize,
        keeping the correct reactant-product atom correspondance.
        
        Args:
            reaction_smiles (str): A reaction SMILES string with mapping numbers.
            
        Returns:
            str: Canonicalized atom-mapped reaction SMILES string.
        """

        all_mols = []
        reactant_mols = []
        for reactant in reaction_smiles.split('>>')[0].split('.'):
            reactant_mols.append(Chem.MolFromSmiles(reactant))
            all_mols.append(Chem.MolFromSmiles(reactant))
        product_mols = []
        for product in reaction_smiles.split('>>')[1].split('.'):
            product_mols.append(Chem.MolFromSmiles(product))
            all_mols.append(Chem.MolFromSmiles(product))
        
        atom_map_dict = {}
        next_map_num = 1
        
        for product_smiles in reaction_smiles.split('>>')[1].split('.'):
            mol = Chem.MolFromSmiles(product_smiles)
            for atom in mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num > 0 and map_num not in atom_map_dict:
                    atom_map_dict[map_num] = next_map_num
                    next_map_num += 1
                    
        for reactant_smiles in reaction_smiles.split('>>')[0].split('.'):
            mol = Chem.MolFromSmiles(reactant_smiles)
            for atom in mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num > 0 and map_num not in atom_map_dict:
                    atom_map_dict[map_num] = next_map_num
                    next_map_num += 1

        for mol in all_mols:
            for atom in mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    atom.SetAtomMapNum(atom_map_dict[map_num])

        for product_mol in product_mols:
            product_mol_copy = Chem.MolFromSmiles(Chem.MolToSmiles(product_mol))
            exact_match = False
            product_smiles = Chem.MolToSmiles(product_mol_copy)
            [atom.SetAtomMapNum(0) for atom in product_mol_copy.GetAtoms()]
            product_smiles_no_mapping = Chem.MolToSmiles(product_mol_copy)
            for i, reactant_mol in enumerate(reactant_mols):
                if not exact_match:
                    reactant_smiles = Chem.MolToSmiles(reactant_mol)
                    if reactant_smiles == product_smiles_no_mapping:
                        reactant_mols[i] = Chem.MolFromSmiles(product_smiles)
                        exact_match = True


        reactant_atom_symbol_freq_dict = {}
        for reactant_mol in reactant_mols:
            seen_canonical_ranks = []
            for reactant_atom, canonical_ranking in zip(reactant_mol.GetAtoms(), Chem.CanonicalRankAtoms(reactant_mol, breakTies=False)):
                if canonical_ranking not in seen_canonical_ranks:
                    if reactant_atom.GetSymbol() not in reactant_atom_symbol_freq_dict:
                        reactant_atom_symbol_freq_dict[reactant_atom.GetSymbol()] = 1
                        seen_canonical_ranks.append(canonical_ranking)
                    else:
                        freq = reactant_atom_symbol_freq_dict[reactant_atom.GetSymbol()] + 1
                        reactant_atom_symbol_freq_dict[reactant_atom.GetSymbol()] = freq
                        seen_canonical_ranks.append(canonical_ranking)
        
        reactant_atom_single_occurance_dict = {k:v for k,v in reactant_atom_symbol_freq_dict.items() if v==1}

        mapped_product_idx = []
        for product_mol in product_mols:
            for product_atom in product_mol.GetAtoms():
                for reactant_mol in reactant_mols:
                    for reactant_atom in reactant_mol.GetAtoms():
                        if product_atom.GetSymbol() == reactant_atom.GetSymbol():
                            if product_atom.GetSymbol() in reactant_atom_single_occurance_dict:
                                if reactant_atom.GetAtomMapNum() == 0 or reactant_atom.GetAtomMapNum() >= 900:
                                    if product_atom.GetAtomMapNum() != 0 and product_atom.GetAtomMapNum() not in mapped_product_idx:
                                        reactant_atom.SetAtomMapNum(product_atom.GetAtomMapNum())
                                        mapped_product_idx.append(product_atom.GetAtomMapNum())
                
        
        product_atoms = {}
        mapped_product_atoms = []
        for product_mol in product_mols: 
            for atom in product_mol.GetAtoms():
                product_atoms[atom.GetAtomMapNum()] = atom.GetSymbol()
                if atom.GetAtomMapNum() != 0:
                    mapped_product_atoms.append(atom.GetAtomMapNum())
                else:
                    print('Error mapping: Unmapped product atoms')
                    return ''

        reactant_atoms = []
        mapped_reactant_atoms = []
        for reactant_mol in reactant_mols: 
            for atom in reactant_mol.GetAtoms():
                reactant_atoms.append(atom.GetAtomMapNum())
                if atom.GetAtomMapNum() != 0:
                    mapped_reactant_atoms.append(atom.GetAtomMapNum())

        if len(mapped_product_atoms) != len(set(mapped_product_atoms)):
            print('Error mapping: Duplicate product atoms')
            return ''
        if len(mapped_reactant_atoms) != len(set(mapped_reactant_atoms)):
            print('Error mapping: Duplicate reactant atoms')
            return ''
    
        seen_reactant_atoms = []
        for reactant_mol in reactant_mols:
            for atom in reactant_mol.GetAtoms():
                if atom.GetAtomMapNum() not in product_atoms:
                    atom.SetAtomMapNum(0)
                else:
                    if atom.GetSymbol() != product_atoms[atom.GetAtomMapNum()]:
                        print('Error mapping: Atomic transmutation!')
                        return ''
                    seen_reactant_atoms.append(atom.GetAtomMapNum())

        if set(seen_reactant_atoms) != set(product_atoms):
            print('Error mapping: Mapped product atoms but not corresponding reactant atoms')
            return ''
    
        reactants_smiles = sorted([Chem.MolToSmiles(mol, canonical=True) for mol in reactant_mols if mol != ''])
        products_smiles = sorted([Chem.MolToSmiles(mol, canonical=True) for mol in product_mols if mol != ''])
        
        canonicalized_rxn = '.'.join(reactants_smiles) + '>>' + '.'.join(products_smiles)
        return canonicalized_rxn

    def _apply_templates(self, templates_list, product):
        """
        Apply reaction templates using rdChiral.
        
        This function takes a list of data from SMARTS patterns and a product SMILES 
        string, and applies each pattern to the product SMILES, adding the resulting
        atom-mapped reactants to a list.
        
        Args:
            templates_list (List): List of data from SMARTS patterns.
            product_smiles (str): The SMILES string of the target product molecule.
            
        Returns:
            List: A list of atom-mapped reactant SMILES strings representing possible reactants.
        """
    
        [product_mol, reactant_mol, rdc_reactants] = product

        rdc_mol = rdc_reactants.reactants
        for atom in rdc_mol.GetAtoms():
            atom.SetAtomMapNum(rdc_reactants.idx_to_mapnum(atom.GetIdx()))
        mapped_product = Chem.MolToSmiles(rdc_mol)

        all_outcomes = []
        applied_smirks = []
        for template in templates_list:
            try:
                products_smarts = template[0]
                reactant_smarts = template[1]
                rdc_rxn = template[2]
                top_smirks = template[3]
                specific_smirks = template[4]
    
                product_mol_has_substruct_match = True
                for smarts_fragment in products_smarts:
                    smarts_fragment_has_match = False
                    for product_fragment in product_mol:
                        if product_fragment.HasSubstructMatch(smarts_fragment):
                            smarts_fragment_has_match = True
                    if not smarts_fragment_has_match:
                        product_mol_has_substruct_match = False
    
                reactant_mol_has_substruct_match = True
                for smarts_fragment in reactant_smarts:
                    smarts_fragment_has_match = False
                    for reactant_fragment in reactant_mol:
                        if reactant_fragment.HasSubstructMatch(smarts_fragment):
                            smarts_fragment_has_match = True
                    if not smarts_fragment_has_match:
                        reactant_mol_has_substruct_match = False
                
                if product_mol_has_substruct_match and reactant_mol_has_substruct_match:
                    outcomes = rdc.rdchiralRun(rdc_rxn, rdc_reactants, return_mapped=True)
                    all_outcomes.append(outcomes)
                    applied_smirks.append(template)
                    
            except:
                pass
    
        return mapped_product, all_outcomes, applied_smirks
    
    def _split_reaction_components(self, reaction_smiles):
        parts = reaction_smiles.strip().split('>>')
        reactants = parts[0]
        products = parts[1]
        return reactants, products
    
    def _prepare_reaction_data(self, reactants, products):
        return [
            [Chem.MolFromSmiles(product) for product in products.split('.')],
            [Chem.MolFromSmiles(reactant) for reactant in reactants.split('.')],
            rdc.rdchiralReactants(products)
        ]
    
    def _process_templates(self, smirks_patterns, reaction_smiles_data, unmapped_reactants, unmapped_products):
        mapped_outcomes = []
        successful_applied_smirks = []
        all_rdc_outcomes = []
        
        atom_mapped_product, rdc_outcomes, applied_smirks = self._apply_templates(smirks_patterns, reaction_smiles_data)
        for rdc_outcome, applied_smirk in zip(rdc_outcomes, applied_smirks):
            mapped_outcomes, successful_applied_smirks = self._process_single_outcome(
                rdc_outcome, 
                applied_smirk, 
                unmapped_reactants, 
                atom_mapped_product, 
                mapped_outcomes, 
                successful_applied_smirks,
                all_rdc_outcomes
            )

        return mapped_outcomes, successful_applied_smirks

    def _process_single_outcome(self, rdc_outcome, applied_smirk, unmapped_reactants, atom_mapped_product, 
                              mapped_outcomes, successful_applied_smirks, all_rdc_outcomes):
        reactants_list, atom_mapped_reactants_dict = rdc_outcome
        for reactant in reactants_list:
            if reactant not in atom_mapped_reactants_dict:
                continue
                
            mapped_outcome = atom_mapped_reactants_dict[reactant][0]
            unmapped_outcome = self._canonicalize_smiles(mapped_outcome)
            all_rdc_outcomes.append(mapped_outcome)
            
            missing_fragments = self._find_missing_fragments(unmapped_outcome, mapped_outcome, unmapped_reactants)
            
            all_fragments_present = len(missing_fragments) == 0
            
            if all_fragments_present:
                mapped_outcomes, successful_applied_smirks = self._handle_complete_fragments(
                    unmapped_reactants, 
                    unmapped_outcome, 
                    mapped_outcome, 
                    atom_mapped_product, 
                    mapped_outcomes, 
                    successful_applied_smirks, 
                    applied_smirk
                )
            else:
                mapped_outcomes, successful_applied_smirks = self._handle_missing_fragments(
                    missing_fragments, 
                    unmapped_reactants, 
                    unmapped_outcome, 
                    mapped_outcome, 
                    atom_mapped_product, 
                    mapped_outcomes, 
                    successful_applied_smirks, 
                    applied_smirk
                )

        return mapped_outcomes, successful_applied_smirks
    
    def _find_missing_fragments(self, unmapped_outcome, mapped_outcome, unmapped_reactants):

        missing_fragments = []
        reactant_fragments = unmapped_reactants.split('.')
        
        for mapped_fragment in mapped_outcome.split('.'):
            unmapped_fragment = self._canonicalize_smiles(mapped_fragment)
            if unmapped_fragment not in reactant_fragments:
                missing_fragments.append([unmapped_fragment, mapped_fragment])
        
        return missing_fragments
    
    def _are_fragments_substructures(self, missing_fragments, unmapped_reactants):
        for unmapped_fragment, _ in missing_fragments:
            try:
                is_substruct = False
                fragment_smarts = Chem.MolFromSmarts(unmapped_fragment)
                
                for reactants_fragment in unmapped_reactants.split('.'):
                    try:
                        if '*' in unmapped_fragment:
                            fragment_reactants = Chem.MolFromSmarts(reactants_fragment)
                            if fragment_reactants.HasSubstructMatch(fragment_smarts):
                                is_substruct = True
                    except:
                        pass
                        
                if not is_substruct:
                    return False
            except:
                pass
        
        return True
    
    def _handle_complete_fragments(self, unmapped_reactants, unmapped_outcome, mapped_outcome, 
                                 atom_mapped_product, mapped_outcomes, successful_applied_smirks, applied_smirk):
        spectators = self._find_spectators(unmapped_reactants, unmapped_outcome)
        spectators_string = '.'.join(spectators)
        reactants_list = [ele for ele in [spectators_string, mapped_outcome] if ele != '']
        finalized_mapped_outcome = '.'.join(reactants_list) + '>>' + atom_mapped_product
        if finalized_mapped_outcome not in mapped_outcomes or applied_smirk[-2] not in [ele[-2] for ele in successful_applied_smirks]:
            mapped_outcomes.append(finalized_mapped_outcome)
            successful_applied_smirks.append(applied_smirk)

        return mapped_outcomes, successful_applied_smirks
    
    def _handle_missing_fragments(self, missing_fragments, unmapped_reactants, unmapped_outcome, mapped_outcome, 
                                atom_mapped_product, mapped_outcomes, successful_applied_smirks, applied_smirk):
        all_fragments_substructs = self._are_fragments_substructures(missing_fragments, unmapped_reactants)
        if all_fragments_substructs:
            identified_mapped_outcomes, identified_unmapped_outcomes = self._identify_and_map_fragments(
                missing_fragments, 
                unmapped_reactants, 
                mapped_outcome, 
                unmapped_outcome
            )

            for mapped_outcome, unmapped_outcome in zip(identified_mapped_outcomes, identified_unmapped_outcomes):
                if mapped_outcome is not None:
                    spectators = self._find_spectators(unmapped_reactants, unmapped_outcome)
                    spectators_string = '.'.join(spectators)
                    reactants_list = [ele for ele in [spectators_string, mapped_outcome] if ele != '']
                    
                    finalized_mapped_outcome = '.'.join(reactants_list) + '>>' + atom_mapped_product
                    
                    if '*' not in finalized_mapped_outcome:
                        if finalized_mapped_outcome not in mapped_outcomes or applied_smirk[-2] not in [ele[-2] for ele in successful_applied_smirks]:
                            mapped_outcomes.append(finalized_mapped_outcome)
                            successful_applied_smirks.append(applied_smirk)

        return mapped_outcomes, successful_applied_smirks
    
    def _find_spectators(self, unmapped_reactants, unmapped_outcome):
        spectators = []
        for fragment in unmapped_reactants.split('.'):
            if fragment not in unmapped_outcome.split('.'):
                spectators.append(fragment)
        return spectators
    
    def _identify_and_map_fragments(self, missing_fragments, unmapped_reactants, original_mapped_outcome, original_unmapped_outcome):
        all_missing_fragments_identified = True

        all_mapped_outcomes = [original_mapped_outcome]
        all_unmapped_outcomes = [original_unmapped_outcome]

        for unmapped_fragment, mapped_fragment in missing_fragments:

            for mapped_outcome, unmapped_outcome in zip(all_mapped_outcomes, all_unmapped_outcomes):
                fragment_found = False
                fragment_mapped_outcomes = []
                fragment_unmapped_outcomes = []
                for fragment in unmapped_reactants.split('.'):
                    out = self._transfer_mapping(mapped_fragment, fragment)

                    if out is not None:
                        fragment_found = True
                        fragment_mapped_outcome = mapped_outcome.replace(mapped_fragment, '')
                        fragment_mapped_outcome = fragment_mapped_outcome.strip('.')
                        fragment_unmapped_outcome = unmapped_outcome.replace(unmapped_fragment, '')
                        fragment_unmapped_outcome = fragment_unmapped_outcome.strip('.')

                        fragment_mapped_outcome += '.' + out
                        fragment_unmapped_outcome += '.' + fragment
                        
                        fragment_mapped_outcome = fragment_mapped_outcome.strip('.')
                        fragment_unmapped_outcome = fragment_unmapped_outcome.strip('.')

                        fragment_mapped_outcomes.append(fragment_mapped_outcome)
                        fragment_unmapped_outcomes.append(fragment_unmapped_outcome)
            
                if not fragment_found:
                    all_missing_fragments_identified = False

            all_mapped_outcomes = fragment_mapped_outcomes
            all_unmapped_outcomes = fragment_unmapped_outcomes

        if all_missing_fragments_identified:
            return all_mapped_outcomes, all_unmapped_outcomes
        else:
            return [], []

    def map_reaction(self, reaction_smiles):
        """
        Maps atoms between reactants and products in a chemical reaction.
    
        This function takes a reaction SMILES string and attempts to create a mapping between
        atoms in the reactants and products using a library of named reactions. It processes
        the reaction using RDKit and RDChiral, assigns stereochemistry, and generates atom mappings.
    
        Args:
            reaction_smiles (str): A SMILES string representing a chemical reaction in the format
                "reactants>>products"
    
        Returns:
            dict: XXX.
    
        Example:
            >>> mapper = ReactionMapper()
            >>> mapped = mapper.map_reaction("CC(=O)O.CN>>CC(=O)NC")
            >>> print(mapped)
            '[CH3:1][C:2](=[O:3])[OH:4].[NH2:5][CH3:6]>>[CH3:1][NH:2][C:3]([CH3:5])=[O:6]'
        """
        reaction_smiles = self._canonicalize_reaction_smiles(reaction_smiles)
        reactants, products = self._split_reaction_components(reaction_smiles)
        
        reaction_smiles_data = self._prepare_reaction_data(reactants, products)
        
        unmapped_reactants = self._canonicalize_smiles(reactants)
        unmapped_products = self._canonicalize_smiles(products)
        
        mapped_outcomes, successful_applied_smirks = self._process_templates(
            self.smirks_patterns, 
            reaction_smiles_data, 
            unmapped_reactants, 
            unmapped_products
        )

        mapped_outcomes = [self._canonicalize_atom_mapping(ele) for ele in list(set(mapped_outcomes))]
        possible_mappings = list(set([ele for ele in mapped_outcomes if ele != '']))
        possible_mappings = [ele for ele in possible_mappings if self._canonicalize_reaction_smiles(ele) == reaction_smiles]

        applied_smirks_names = []
        for applied_smirk_data in successful_applied_smirks:
            applied_smirk = applied_smirk_data[-2]
            applied_smirk_forward = applied_smirk.split('>>')[1] + '>>' + applied_smirk.split('>>')[0]
            applied_smirks_names.append(self.smirks_name_dictionary[applied_smirk_forward])
            
        
        if len(possible_mappings) == 1:
            mapping_dict = {'mapping': possible_mappings[0], 'reaction_classification': applied_smirks_names}
            return mapping_dict

        return {'mapping': '', 'reaction_classification': []}
