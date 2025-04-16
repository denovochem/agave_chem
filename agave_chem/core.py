from rdkit import Chem
from rdkit.Chem import AllChem
from rdchiral import main as rdc
import json
from ast import literal_eval
import re

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
        self.smirks_patterns = self._initialize_template_data(smirks_patterns)

    def _expand_first_bracketed_list(self, input_string):
        """
        Finds the first bracketed list with commas (e.g., [A,B]) and expands it.
    
        Args:
            input_string: The string to process.
    
        Returns:
            A list of strings, where each string represents one expansion
            of the first found bracketed list. If no such list is found,
            returns a list containing just the original string.
        """
        # Regex to find the first occurrence of [...] containing a comma
        # It captures:
        # group(0): The whole match including brackets (e.g., "[Li,Na]")
        # group(1): The content inside the brackets (e.g., "Li,Na")
        match = re.search(r'(\[([^\]]*?,[^\]]*?)\])', input_string)
    
        if not match:
            return [input_string]
    
        original_part = match.group(1) 
        content = match.group(2)      
        elements = content.split(',')  
    
        results = []
        for element in elements:
            new_string = input_string.replace(original_part, f'[{element}]', 1)
            results.append(new_string)
    
        return results
    
    def _expand_all_recursively(self, input_string):
        """
        Recursively applies expand_first_bracketed_list to a string until
        no more expansions with commas are possible, collecting all final combinations.
    
        Args:
            input_string: The string to start the expansion from.
    
        Returns:
            A list of all fully expanded strings.
        """
        expanded_list = self._expand_first_bracketed_list(input_string)
    
        # Base Case: If the expansion didn't change the string
        # (meaning no comma-list was found), we've reached a final state for this branch.
        if len(expanded_list) == 1 and expanded_list[0] == input_string:
            return expanded_list 
    
        # Recursive Step: If expansion occurred, apply the function recursively
        # to each of the newly generated strings and collect the results.
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
        """
        all_smirks = []
        for reaction in named_reactions:
            # Reverse the SMIRKS pattern (products >> reactants becomes reactants >> products)
            smirks = reaction['smirks'].split('>>')[1] + '>>' + reaction['smirks'].split('>>')[0]
            
            # Only include patterns that don't expand to too many variations
            if len(self._expand_all_recursively(smirks)) < 100:
                all_smirks.extend(self._expand_all_recursively(smirks))
        
        rdc_info = []
        for smirk in all_smirks:
            try:
                products_smarts = [Chem.MolFromSmarts(ele) for ele in smirk.split('>>')[0].split('.')]
                reactants_smarts = [Chem.MolFromSmarts(ele) for ele in smirk.split('>>')[1].split('.')]
                
                rdc_rxn = rdc.rdchiralReaction(smirk)
                
                rdc_info.append([
                    products_smarts, 
                    reactants_smarts,
                    rdc_rxn, 
                ])
            except:
                pass
        
        return rdc_info

    def _remove_mapping_from_smiles(self, smiles):
        """
        Removes atom mapping numbers from a SMILES string and returns a canonicalized version.
        
        This function splits a SMILES string by '.' (which separates disconnected structures),
        removes all atom mapping numbers, canonicalizes each component, sorts them,
        and rejoins them with '.' separators.
        
        Args:
            smiles (str): A SMILES string, potentially with atom mapping numbers.
            
        Returns:
            str: Canonicalized SMILES string with atom mapping numbers removed.
        """
        mols = [Chem.MolFromSmiles(ele) for ele in smiles.split('.')]
        
        for mol in mols:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
        
        canonicalized_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in mols]
        
        canonicalized_smiles = sorted(canonicalized_smiles)
        
        return '.'.join(canonicalized_smiles)

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
        rxn = AllChem.ReactionFromSmarts(reaction_smiles, useSmiles=True)
        
        all_mols = []
        for reactant in rxn.GetReactants():
            all_mols.append(reactant)
        for product in rxn.GetProducts():
            all_mols.append(product)
        
        atom_map_dict = {}
        next_map_num = 1
        
        for mol in all_mols:
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
        
        reactants = [Chem.MolToSmiles(mol) for mol in rxn.GetReactants()]
        products = [Chem.MolToSmiles(mol) for mol in rxn.GetProducts()]
        
        canonicalized_rxn = '.'.join(reactants) + '>>' + '.'.join(products)
        
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

        all_outcomes = []
        for template in templates_list:
            try:
                products_smarts = template[0]
                reactant_smarts = template[1]
                rdc_rxn = template[2]
    
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
            except:
                pass
    
        return all_outcomes

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
            str: A reaction SMILES string with atom mapping numbers if successful mapping is found.
                Returns an empty string if multiple possible mappings exist or if mapping fails.
    
        Example:
            >>> mapper = ReactionMapper()
            >>> mapped = mapper.map_reaction("CC(=O)O.CN>>CC(=O)NC")
            >>> print(mapped)
            '[CH3:1][C:2](=[O:3])[OH:4].[NH2:5][CH3:6]>>[CH3:1][NH:2][C:3]([CH3:5])=[O:6]'
        """
        reactants = reaction_smiles.strip().split('>>')[0]
        products = reaction_smiles.strip().split('>>')[1]
    
        reaction_smiles_data = [
            [Chem.MolFromSmiles(product) for product in products.split('.')],
            [Chem.MolFromSmiles(reactant) for reactant in reactants.split('.')],
            rdc.rdchiralReactants(products)
        ]
    
        unmapped_reactants = self._remove_mapping_from_smiles(reactants)
        unmapped_products = self._remove_mapping_from_smiles(products)
    
        product_mol = Chem.MolFromSmiles(unmapped_products)
        Chem.AssignStereochemistry(product_mol, flagPossibleStereoCenters=True)
        product_mol.UpdatePropertyCache(strict=False)
    
        for atom_num, atom in enumerate(product_mol.GetAtoms()):
            atom.SetAtomMapNum(atom_num+1)
        atom_mapped_product = Chem.MolToSmiles(product_mol)
    
        mapped_outcomes = []
        rdc_outcomes = self._apply_templates(self.smirks_patterns, reaction_smiles_data)
        for rdc_outcome in rdc_outcomes:
            [reactants_list, atom_mapped_reactants_dict] = rdc_outcome
            for reactant in reactants_list:
                if reactant in atom_mapped_reactants_dict:
                    mapped_outcome = atom_mapped_reactants_dict[reactant][0]
                    unmapped_outcome = self._remove_mapping_from_smiles(mapped_outcome)
                    all_fragments_present = True
                    for fragment in unmapped_outcome.split('.'):
                        if fragment not in unmapped_reactants.split('.'):
                            all_fragments_present = False
                    if all_fragments_present:
                        mapped_outcomes.append(mapped_outcome + '>>' + atom_mapped_product)
    
        possible_mappings = list(set(mapped_outcomes))
        if len(possible_mappings) == 1:
            return self._canonicalize_atom_mapping(mapped_outcomes[0])
        return ''
