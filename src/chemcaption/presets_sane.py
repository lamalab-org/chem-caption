# -*- coding: utf-8 -*-

"""Presets for SMARTS and molecular substructure matching."""

from typing import Dict

__all__ = [
    "CORE",
    "BRANCHES",
    "HOMOAROMATICS",
    "HETEROAROMATICS",
    "ALIPHATIC_RINGS",
    "HETEROALIPHATIC_RINGS",
    "OXO_RINGS",
    "MAIN_GROUP",
    "BIOMOLECULES",
]

CORE: Dict[str, str] = {
    # Carbon (excluding aromatics)
    "isolated_C=C": "[!$([#6,O,N,S]:,=,#[#6,O,N,S])]-C(-[!$([#6,O,N,S]:,=,#[#6,O,N,S])])=C(-[!$([#6,O,N,S]:,=,#[#6,O,N,S])])-[!$([#6,O,N,S]:,=,#[#6,O,N,S])]",
    "conjugated_C=C": "[*]~[C;!$([#6;X3]1~[#6;X3]~[#6;X3]~[#6;X3]~[#7,#8,#16]~1)](~[*])=[C;!$([#6;X3]1~[#6;X3]~[#6;X3]~[#6;X3]~[#7,#8,#16]~1)](~[*])-[$([#6]:,=,#[#6])]", # ~[*] are added for correct counting
    "heteroconjugated_C=C": "[*]~[C;!$([#6;X3]1~[#6;X3]~[#6;X3]~[#6;X3]~[#7,#8,#16]~1)](~[*])=[C;!$([#6;X3]1~[#6;X3]~[#6;X3]~[#6;X3]~[#7,#8,#16]~1)](~[*])-[$([C,O,N,S]=,#[C,O,N,S]);!$(C=,#C)]",
    "conjugated_C#C": "[*]~C(~[*])#C(~[*])-[$(C=,#C)]",
    "terminal_alkyne": "[CH0]#[CH]",
    "nonterminal_alkyne": "[CH0]#[CH0]",
    "allene": "[CX3]=C=[CX3]",
    "enyne": "[!$(C#C)]-C(-[!$(C#C)])=C(-[!$(C#C)])-C#C",
    "enediyne": "C#C-C(-[!$(C#C)])=C(-[!$(C#C)])-C#C",
    "carbene": "[*]-[CX2]-[*]",
    # Oxygen
    ## 1x O
    "primary_alcohol": "[CX4;H2]-[O-,OH,OH2+]",
    "secondary_alcohol": "[CX4;H;!$(C(O)(O))](-[O-,OH,OH2+])",
    "tertiary_alcohol": "[CX4;H0;!$(C(O)(O))](-[O-,OH,OH2+])",
    "aromatic_hydroxy": "[cX3H0]-[O-,OH,OH2+]", 
    "geminal_diol": "[CX4](-[O-,OH,OH2+])-[O-,OH,OH2+]",
    "vicinal_diol": "[OH]-[CX4]-[CX4]-[OH]",
    "enol": "[#6]=C-[OH,O-,OH2+]",
    "ether": "[#6;!$(C#C);!$(C=[C,O,S,N]);!$(C(-[O,S,N,n])-[O,S,N,n]);!$(C#N)]-[OX2;!r3;!$(O1-[#6;X3]~[#6;X3]~[O,S,N]~[#6;X3]~[#6;X3]-1)]-[#6;!$(C#C);!$(C=[C,O,S,N]);!$(C([O,S,N,n])-[O,S,N,n]);!$(C#N)]", # excludes enol ethers, ynol ethers, epoxides, acetals, esters, anhydrides, oxines, cyanates
    "enol_ether": "[CX3;$(C=C);!$(C(O)O)]-O-[#6;!$(C=[O,S,N]);!$(C#N)]",
    "ynol_ether": "[CX2;$(C#C)]-O-[#6;!$(C=[O,S,N]);!$(C#N)]",
    "aldehyde": "[#6]-[CH](=O)",
    "ketone": "[#6]-C(=O)-[#6]",
    "ketene": "C=C=O",
    "conjugated_carbonyl": "O=[C;$(C-C=C)]", # 'O=C-C=C' is not used; it would double count the carbonyl if it's conjugated on both sides
    "oxonium": "[!#1]-[OX3+](-[!#1])-[!#1]",
    "acetyl": "[!$([CH2](-C(=O)-[CH3])-C=O)]-C(=O)-[CH3]", # excludes acetoxy, acetoacetyl
    "other_acyl": "[#6,#1]-C(=O)-[!$([#6,F,Cl,Br,I]);!$(O-[#6,#7,#8]);!$(S-[#6]);!$([#7;X3]);!$(N=C)]",
    ## 2x O
    "methylenedioxy": "[*]-O-[CH2;!$([CX4]1Oc2ccccc2O1)]-O-[*]", # excludes benzodioxoles
    "peroxide": "[!#1]-O-O-[!#1]",
    "hydroperoxy": "[!#1]-O-[O-,OH,OH2+]",
    "aliphatic_carboxylic_acid": "C-C(=O)-[OH,O-,OH2+]",
    "aromatic_carboxylic_acid": "c-C(=O)-[OH,O-,OH2+]",
    "carboxylate_ester": "[#6,#1]-C(=O)-O-[#6;!$(C=[O,S])]",
    "hemiacetal": "[O-,OH,OH2+]-[CX4;!$(C(O)(O)[N,O])]-O-[!#1;!$(C=O)]", # including hemiketals
    "acetal": "[!#1;!$(C=O)]-O-[CX4;!$(C(O)(O)[N,O]);!H2;!$([CX4]1Oc2ccccc2O1)]-O-[!#1;!$(C=O)]", # including ketals, excluding acylals, methylenedioxy
    "ketene_acetal": "C=C(-O)-O",
    "acetoxy": "[CH3]-C(=O)-O-[!#1]",
    "acetylenediolate": "[!#1]-O-C#C-O-[!#1]",
    ### common diacyls
    "oxalyl": "[!#6;!#1]-C(=O)-C(=O)-[!#6;!#1]",
    "acetylenedicarboxoyl": "[!#6;!#1]-C(=O)-C#C-C(=O)-[!#6;!#1]",
    "malonyl": "[!#6;!#1]-C(=O)-[CX4]-C(=O)-[!#6;!#1]",
    "acetoacetyl": "[CH3]-C(=O)-[CX4]-C(=O)-[!#6;!#1]",
    "pyruvyl": "[CH3]-C(=O)-C(=O)-[!#6;!#1]",
    "succinyl": "[$(C(=O)-[!#6;!#1])]-[C;!$(C-N)]-[C;!$(C-N)]-[$(C(=O)-[!#6;!#1])]", # excludes aspartate
    "glutaryl": "[$(C(=O)-[!#6;!#1])]-[C;!$(C-N)]-C-[C;!$(C-N)]-[$(C(=O)-[!#6;!#1])]", # excludes glutamate
    "adipoyl": "[$(C(=O)-[!#6;!#1])]-C-C-C-C-[$(C(=O)-[!#6;!#1])]",
    "maleoyl": "[$(C(=O)-[!#6;!#1])]-\[CX3]=[CX3]/-[$(C(=O)-[!#6;!#1])]",
    "phthaloyl": "[$(C(=O)-[!#6;!#1])]-c1ccccc1-C(=O)-[$(C(=O)-[!#6;!#1])]",
    "isophthaloyl": "[$(C(=O)-[!#6;!#1])]-c1cc-[$(C(=O)-[!#6;!#1])]ccc1",
    "terephthaloyl": "[$(C(=O)-[!#6;!#1])]-c1ccc-[$(C(=O)-[!#6;!#1])]cc1",
    ## 3x O
    "percarboxylic_acid": "[#6]-C(=O)-O-[O-,OH,OH2+]",
    "percarboxylate_ester": "[#6]-C(=O)-O-O-[#6;!$(C=O)]",
    "carbonate": "[*]-O-C(=O)-O-[*]",
    "carboxylic_anhydride": "[*]-C(=O)-O-C(=O)-[*]",
    "orthoester": "[#6,#1]-C(-O-[#6,#14;!$(C=[O,S,N])])(-O-[#6,#14;!$(C=[O,S,N])])-O-[#6,#14;!$(C=[O,S,N])]",
    "ozonide": "O1-[CX4]-O-O-[CX4]-1",
    "alpha-keto_acid": "[#6,#1]-C(=O)-C(=O)-[OH,O-,OH2+]",
    "alpha-keto_ester": "[#6,#1]-C(=O)-C(=O)-O-[#6,#14;!$(C=[O,S])]",
    "hemiacylal": "C(=O)-O-[CX4]-[O-,OH,OH2+]",
    "acyl_hemiacetal": "C(=O)-O-[CX4]-O-[#6;!$(C=O)]",
    "deltate": "O=c1c(O)c1(O)",
    "squarate": "O=c1c(=O)c(O)c1O",
    ## 4+ O
    "diacyl_peroxide": "[#6]-C(=O)-O-O-C(=O)-[#6]",
    "orthocarbonate": "O-[CX4](-O)(-O)-O",
    "acylal": "C(=O)-O-[CX4]-O-C(=O)",

    # Nitrogen
    ## 1x N
    "primary_amine": "[NX3H2,NX4H3+]-[#6;!$(C=[O,S,N])]", # excludes anillines, amides, etc.
    "aryl_amine": "[$(N-[c;!r3;!r4](:[!$(c=O)]):[!$(c=O)]);!$(N=[O,S,N,P])]",
    "secondary_amine": "[#6;!$(C=[O,S,N])]-[NX3H,NX4H2+;!r3;!$(N1~[#6;X3]~[#6;X3]~[O,S]~[#6;X3]~[#6;X3]-1)]-[#6;!$(C=[O,S,N])]", # excludes amides, aziridines, oxazines, thiazines
    "tertiary_amine": "[#6;!$(C=[O,S,N])]-[NX3H0,NX4H+;!r3;!$(N1~[#6;X3]~[#6;X3]~[O,S]~[#6;X3]~[#6;X3]-1)](-[#6;!$(C=[O,S,N])])-[#6;!$(C=[O,S,N])]", # excludes amides, aziridines, oxazines, thiazines
    "quaternary_ammonium": "[!$([#6,#8;-])]-[NX4H0+](-[!$([#6,#8;-])])(-[!$([#6,#8;-])])-[!$([#6,#8;-])]", # excludes N-oxides and ammonium ylides
    "ammonium_ylide": "[#6-]-[NX4H0+](-[#6;!-])(-[#6;!-])-[#6;!-]",
    "imine": "[#6,#1]-C(=[NX2,NH2+,NX3H+;!$(N-[O,SX2,N]);!r3])-[#6,#1]", # excludes azirines, amidines, guanidines, carbodiimides, isoureas
    "ketenimine": "[#6,#1]-C(=C=[NX2,NH2+,NX3H+])-[#6,#1]",
    "iminium": "[CX3]=[NX3+;H0;!$([NX3+]-O)]", # excludes nitrones, nitronates
    "enamine": "C=[C;!$(C(-N)-N)]-[NX3,NX4H+,NX4H2+,NH3+;!$(N1~[#6;X3]~[#6;X3]~[O,S]~[#6;X3]~[#6;X3]-1)](-[#6,#1;!$(C=[O,S,N])])-[#6,#1;!$(C=[O,S,N])]", # excludes enediamines, oxazines, thiazines
    "nitrile": "[#6,#1]-[CX2]#[NX1H0,NX2H+]", # excludes cyanates, cyanamides
    "isonitrile": "[#6,#1]-[NX2H0+]#[CX1-]",
    "azomethine_ylide": "[#6,#14,#1]-[CX3-](-[#6,#14,#1])-[NX3+](-[#6,#14,#1])=[CX2](-[#6,#14,#1])-[#6,#14,#1]",
    ## 2x N
    "hydrazine": "[!$(C=[O,N])]-[NX3,NX4H+,NX4H2+,NH3+](-[!$(C=[O,N])])-[NX3,NX4H+,NX4H2+,NH3+](-[!$(C=[O,N])])-[!$(C=[O,N])]",
    "hydrazone": "[#6,#1]-C(=[NX2,NX3H+;!r5]-[NX3,NX4+])-[!#7;!#8]", # excludes pyrazolines, amidrazones, N-amino imidates
    "aldazine": "[CX3H]=[NX2,NX3H+]-[NX2,NX3H+]=[CX3H]",
    "ketazine": "[CX3H0]=[NX2,NX3H+]-[NX2,NX3H+]=[CX3H0]",
    "azine_N-oxide": "[CX3]=[NX3+](-[O-])-[NX2]=[CX3]",
    "azo": "[*]-[NX2,NX3H+;!$([N+]-[O-])]=[NX2,NX3H+;!$([N+]-[O-])]-[*]", # excludes azodicarboxylates
    "diazo": "C=[NX2+]=[NX1-]",
    "diazonium": "[#6]-[NX2+]#[NX1]",
    "azomethine_imide": "[#6,#14,#1]-[CX3]-,=[NX3+;$(N(=C)(-[N-])),$(N(-[C-])=N)](-[#6,#14,#1])-,=[NX2]-[#6,#14,#1]",
    "amidine": "[#6,#1]-C(=[NX2,NX3H+]-[!O;!N])-[NX3](-[!N])-[!N]",
    "aminal": "[#7]-[CX4;!$(C(N)(N)[O,N])]-[#7]",
    "ketene_aminal": "C=C(-[NX3,NX4+])-[NX3,NX4+]",
    "carbodiimide": "[*]-[NX2,NX3H+]=C=[NX2,NX3H+]-[*]",
    "cyanamide": "[#7;!X4]-[CX2]#[NX1H0,NX2H+]",
    ## 3+ N
    "azide": "[!#7]-[$([NX2]=[NX2+]=[NX1-]),$([NX2-]-[NX2+]#[NX1])]",
    "triazene": "[#6,#14,#1]-N=N-N(-[#6,#14,#1])-[#6,#14,#1]",
    "guanidine": "[NX3]-C(=[NX2,NX3H+,NH2+])-[NX3]",
    "amidrazone": "[#6,#1]-[CX3](-,=N-[NX3])-,=N",
    "orthoamide": "[#7]-[CX4;!$(C([#7])([#7])([#7])[#7])](-[#7])-[#7]",
    "tetraamino_methane": "[#7]-[CX4](-[#7])(-[#7])-[#7]",

    # Oxygen + Nitrogen
    ## 2 hetero atoms
    "cyanohydrin": "O-[CX4]-C#[NX1]",
    "1,2-amino_alcohol": "N-[CX4]-[CX4]-[OH]",
    "hemiaminal": "[#7]-[CX4;!$(C(N)(O)[N,O])]-[OH]",
    "O,N-acetal": "[#7]-[CX4;!$(C(N)(O)[N,O])]-O-[#6,#14]",
    "ketene_N,O-acetal": "C=C(-[#7])-O-[#6,#14]",
    "hydroxylamine": "[O-,OH,OH2+]-[NX3,NX4H+,NX4H2+,NH3+](-[#6,#1;!$(C=[O,S,N])])-[#6,#1;!$(C=[O,S,N])]",
    "O-organyl_hydroxylamine": "[#6,#14;!$(C=[O,S,N])]-[O;!$(O1-N~[#6;X3]~[#6,#7;X3]~[#6,#7;X3]~[#6;X3]-1)]-[NX3,NX4H+,NX4H2+,NH3+](-[#6,#1;!$(C=[O,S,N])])-[#6,#1;!$(C=[O,S,N])]", # excludes hydroxamate esters, amidines, imidates, oxadiazines
    "aminoxyl_radical": "[OX1H0;!-]-[NX3;!$(N=*)]",
    "oxime": "[#6,#1]-C(=[NX2,NX3H+]-O)-[!#7;!#8]", # excludes amidoximes, N-oxy imidates
    "nitroso": "[!#7;!#8]-[NX2]=O",
    "nitrone": "[!#8;!#7]-[NX3+]([O-])=C(-[#6,#1])-[#6,#1]",
    "N-oxoammonium": "O=[NX3+](-[#6])-[#6]",
    "N-oxide": "[O-]-[NX4+]",
    "aromatic_N-oxide": "[O-]-[nX3+](:a):a",
    "nitrile_oxide": "[*]-[CX2]#[NX2+]-[O-]",
    "fulminate": "[*]-O-[NX2+]#[CX1-]",
    "cyanate": "[*]-O-C#N",
    "isocyanate": "[*]-[NX2]=C=O",
    "primary_amide": "[#6,#1]-C(=O)-[NH2]",
    "secondary_amide": "[#6,#1]-C(=O)-[NH]-[#6,#14;!$(C=[O,S])]",
    "tertiary_amide": "[#6,#1]-C(=O)-[#7;X3]([#6,#14;!$(C=[O,S])])[#6,#14;!$(C=[O,S])]", # includes pyrrolide amides
    "acyl_imine": "[#6,#1]-C(=O)-[NX2]=[#6]",
    "imidate": "[#6,#1]-C(=[NX2,NX3H+]-[!#8;!#7])-O-[*]",
    ## 3 hetero atoms
    "amide_hemiacetal": "[#7]-[CX4](-[OH])-O-[#6,#14]",
    "amide_acetal": "[#7]-[CX4](-O-[#6,#14])-O-[#6,#14]",
    "ester_aminal": "[#7]-[CX4](-O-[#6,#14])-[#7]",
    "O-acyl_hydroxylamine": "[#6,#1]-C(=O)-O-[NX3,NX4H+,NX4H2+,NH3+](-[#6,#1;!$(C=[O,S,N])])-[#6,#1;!$(C=[O,S,N])]",
    "amidoxime": "[#6,#1]-C(=[NX2,NX3H+]-O-[#6,#14,#1])-[NX3]",
    "hydroxamic_acid": "[#6,#1]-C(=O)-[NX3](-[OH,O-,OH2+])-[#6,#14,#1;!$(C=[O,S,N])]",
    "hydroxamate_ester": "[#6,#1]-C(=O)-[NH]-O-[#6,#14]",
    "Weinreb_amide": "[#6,#1]-C(=O)-[NX3](-[#6;!$(C=[S,O,N])])-O-[#6;!$(C=[S,O,N])]",
    "hydrazide": "C(=O)-[NX3]-[$([#7;X3,$([NX2]=[CX3])])]",
    "nitro": "[NX3+](=O)([O-])-[!O;!N;!$([cH0]1[cH0]c(-[N+](=O)[O-])[cH]c(-[N+](=O)[O-])[cH]1);!$([cH0]1[cH]c(-[N+](=O)[O-])[cH0]c(-[N+](=O)[O-])[cH]1)]",
    "nitrite": "[*]-O-[NX2]=O",
    "nitrosamine": "[#7]-[NX2]=O",
    "imide": "[$(C(=O)-[#6,#1])]-N(-[!$(C=O)])-[$(C(=O)-[#6,#1])]",
    "urea": "[#7X3,$([NX2]=[CX3])]-C(=[O;!$(O=C1[#7]~[#6]C(=O)N1)])-[NX3,$([NX2]=C)]", # excludes hydantoin
    "isourea": "[NX3]-C(=[NX2,NX3H+])-O-[#6,#14,#1]",
    "carbamate": "[#6]-O-C(=O)-[NX3]",
    "imidocarbonate": "[#6,#14]-O-C(=[NX2,NX3H+])-O-[#6,#14]",
    "N-amino_imidate": "[#6,#1]-C(=[NX2,NX3H+;!$([NX2+]=[NX1-])]-[NX3,NX4+])-O-[*]",
    "N-oxy_imidate": "[#6,#1]-C(=[NX2,NX3H+;!$(N1OC=,:COC=,:1)]-O-[*])-O-[*]",
    "nitronate": "[O-,OX2]-[NX3+]([O-,OX2])=C(-[#6,#1])-[#6,#1]",
    "azoxy": "[#6]-[N+]([O-])=[N,NH+;!$([N+]-[O-])]-[#6]",
    "deltic_monoamide": "O=c1c(O)c1([N;!+])",
    "deltamide": "O=c1c([N;!+])c1([N;!+])",
    ## 4+ hetero atoms
    "acyl_hydroxamate": "[#6,#1]-C(=O)-[NX3]-O-C(=O)-[#6,#1]",
    "nitramine": "[#7]-[NX3+](=O)[O-]",
    "nitrate": "[*]-O-[NX3+](=O)[O-]",
    "azodioxy": "[#6]-[N+]([O-])=[N+]([O-])-[#6]",
    "triacylamine":  "[#6,#1]-C(=O)-N(-C(=O)-[#6,#1])-C(=O)-[#6,#1]",
    "azodicarboxylate": "OC(=O)-[NX2,NX3H+;!$([N+]-[O-])]=[NX2,NX3H+;!$([N+]-[O-])]-C(=O)O",
    "squaric_monoamide": "O=c1c(=O)c(O)c1[N;!+]",
    "squaramide": "O=c1c(=O)c([N;!+])c1([N;!+])",
}

BRANCHES: Dict[str, str] = {
    # aliphatic
    # the (acyclic) aliphatic side-chains are limited to be connected to: non-carbon atoms, non-sp3 carbons atoms, or sp3 carbon atoms in a ring: [!C,$([C;!X4,R])
    # non-branched aliphatic chains are also allowed to be connected a to a quaternary carbon 
    # [!#1] is used because explicit hydrogens are added to the SMILES prior to counting
    ## saturated
    "methyl": "[CH3]-[!#1;!O;!C,$([C;!X4,R,X4H0;!$(C(=O));!$(C(-[CH3])(-[CH3])(-[CH3,$([CH2]-[CH3])]));!$(C(-[CH3])(-[CH3])=[CH]-[CH2]);!$(C(-[CH3])(-[CH2]-[CH2]-[CH]=C(-[CH3])-[CH3])=[CH]-[CH2])]);!$([cH0]1[cH][cH][cH][cH][cH0]1);!$([cH0]1[cH][cH][cH0][cH][cH]1);!$([cH0]1[cH][cH0][cH][cH][cH]1);!$([cH0]1[cH0]c([CH3])[cH]c([CH3])[cH]1);!$([cH0]1[cH]c([CH3])[cH0]c([CH3])[cH]1);!$([SX4](=O)(=O));!$([SX4+](-[O-])(=O));!$([SX4+2](-[O-])(-[O-]));!$([Si]([CH3])([CH3])[CH3]);!$([Si]([CH3])([CH3])C([CH3])([CH3])[CH3]);!$([Si]([CH3])([CH3])[CH]([CH3])[CH3]);!$([Si]([CH3])([CH3])[cH0]1[cH][cH][cH][cH][cH]1)]",
    ##!! "methyl" excludes: methoxy; methoxymethyl; acetoxy; acetyl; methyls on t-butyl, t-pentyl, neopentyl, thexyl, prenyl, geranyl; tolyls; mesyl; mesityl; methyls on TMS, TBDMS, etc.;
    "methoxy": "[CH3]-[O;!$(O(-[CH2])-[CH3])]-[!$(C=O)]", # excludes methoxymethyl, methoxycarbonyl
    "methoxymethyl": "[CH3]-O-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]",
    "carbomethoxy": "[CH3]-O-C(=O)-[*]",
    "ethyl": "[CH3]-[CH2]-[!C,$([C;!X4,R,X4H0;!$(C(-[CH3])(-[CH3]))]);!$([Si]([CH2][CH3])([CH2][CH3])[CH2][CH3]);!O;!#1]", # excludes ethoxy, ethyls on TES and the ethyl on t-pentyl
    "ethoxy": "[CH3]-[CH2]-O-[!$(C=O)]", # excludes ethoxycarbonyl
    "carboethoxy": "[CH3]-[CH2]-O-C(=O)-[*]",
    "n-propyl": "[CH3]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!O;!#1]", # excludes propoxy
    "propoxy": "[CH3]-[CH2]-[CH2]-O-[*]",
    "isopropyl": "[CH3]-[CH](-[CH3])-[!C,$([C;!X4,R,X4H0]);!O;!#1;!$([Si]([CH]([CH3])[CH3])([CH]([CH3])[CH3])([CH]([CH3])[CH3]));!$([Si]([CH]([CH3])[CH3])([CH3])([CH3]));!$([cH0]1[cH0]c(-[CH]([CH3])[CH3])[cH]c(-[$([CH]([CH3])[CH3]),#1])[cH]1);!$([cH0]1[cH]c(-[CH]([CH3])[CH3])[cH0]c(-[CH]([CH3])[CH3])[cH]1)]", # excludes isopropoxy, 2,6-di and 2,4,6-triisopropylphenyl, isopropyls on common silyls
    "isopropoxy": "[CH3]-[CH](-[CH3])-O-[*]",
    "n-butyl": "[CH3]-[CH2]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]",
    "isobutyl": "[CH3]-[CH](-[CH3])-[CH2]-[!C,$([C;!X4,R]);!#1]",
    "s-butyl": "[CH3]-[CH2]-[CH](-[CH3])-[!C,$([C;!X4,R]);!#1]",
    "t-butyl": "[CH3]-C(-[CH3])(-[CH3])-[!C,$([C;!X4,R]);!O;!#1;!$([Si]([CH3])([CH3])C([CH3])([CH3])[CH3]);!$([Si]([cH0]1[cH][cH][cH][cH][cH]1)([cH0]1[cH][cH][cH][cH][cH]1)C([CH3])([CH3])[CH3]);!$([cH0]1[cH0]c(-C([CH3])([CH3])[CH3])[cH]c(-[$(C([CH3])([CH3])[CH3]),#1])[cH]1);!$([cH0]1[cH]c(-C([CH3])([CH3])[CH3])[cH0]c(-C([CH3])([CH3])[CH3])[cH]1)]", # excludes t-butoxy, t-Boc, t-butyls on TBDMS and TBDPS, 2,6-di and 2,4,6-tri-tert-butylphenyl
    "t-butoxy": "[CH3]-C(-[CH3])(-[CH3])-O-[!$(C(=O)(O)[!#6])]", # excludes tBoc
    "t-Boc": "[CH3]-C(-[CH3])(-[CH3])-O-C(=O)-[!#6]",
    "Fmoc": "[cH0]12[cH][cH][cH][cH][cH0]2-[cH0]2[cH][cH][cH][cH][cH0]2-[CH]1-[CH2]-O-C(=O)-[!#6]",
    "n-pentyl": "[CH3]-[CH2]-[CH2]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]",
    "t-pentyl": "[CH3]-[CH2]-C(-[CH3])(-[CH3])-[!C,$([C;!X4,R]);!#1]",
    "isoamyl": "[CH3]-[CH](-[CH3])-[CH2]-[CH2]-[!C,$([C;!X4,R]);!#1]",
    "neopentyl": "[CH3]-C(-[CH3])(-[CH3])-[CH2]-[!C,$([C;!X4,R]);!#1]",
    "n-hexyl": "[CH3]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]",
    "thexyl": "[CH3]-[CH](-[CH3])-C(-[CH3])(-[CH3])-[!C,$([C;!X4,R,X4H0]);!#1]",
    "n-octyl":  "[CH3]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]",
    "lauryl": "[CH3]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]", # dodecyl
    "cetyl":  "[CH3]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]", # hexadecyl
    "stearyl": "[CH3]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]", # octadecyl
    # cyclic
    "cyclopropyl": "[!#1]-[CX4;H1]1-[CH2]-[CH2]-1",
    "cyclobutyl":  "[!#1]-[CX4;H1]1-[CH2]-[CH2]-[CH2]-1",
    "cyclopentyl":  "[!#1]-[CX4;H1]1-[CH2]-[CH2]-[CH2]-[CH2]-1",
    "cyclohexyl":  "[!#1]-[CX4;H1]1-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-1",
    "1-adamantyl": "[!#1]-[CX4;H0]1(-[CH2]2)-[CH2]-[CH](-[CH2]3)-[CH2]-[CH](-[CH2]-1)-[CH2]-[CH]-2-3",
    "2-adamantyl": "[CH]1(-[CH2]2)-[CX4;H1](-[!#1])-[CH](-[CH2]3)-[CH2]-[CH](-[CH2]-1)-[CH2]-[CH]-2-3",
    ## unsaturated / isoprenoid
    "vinyl": "[CH2]=[CH]-[!C,$([C;R]);!#1]",
    "vinylidene": "[CH2]=[CX3H0]",
    "allyl": "[CH2]=[CH]-[CH2]-[!C,$([C;R]);!#1]",
    "propargyl": "[CH]#C-[CH2]-[!C,$([C;R]);!#1]",
    "prenyl": "[CH3]-C(-[CH3])=[CH]-[CH2;!$(C-[CH2]-C(-[CH3])=[CH]-[CH2])]", # excludes geranyl
    "geranyl": "[CH3]-C(-[CH3])=[CH]-[CH2]-[CH2]-C(-[CH3])=[CH]-[CH2;!$(C-[CH2]-C(-[CH3])=[CH]-[CH2])]", # excludes farnesyl
    "farnesyl": "[CH3]-C(-[CH3])=[CH]-[CH2]-[CH2]-C(-[CH3])=[CH]-[CH2]-[CH2]-C(-[CH3])=[CH]-[CH2;!$(C-[CH2]-C(-[CH3])=[CH]-[CH2])]", # excludes geranylgeranyl
    "geranylgeranyl": "[CH3]-C(-[CH3])=[CH]-[CH2]-[CH2]-C(-[CH3])=[CH]-[CH2]-[CH2]-C(-[CH3])=[CH]-[CH2]-[CH2]-C(-[CH3])=[CH]-[CH2]",
    "phytanyl": "[CH3]-[CH](-[CH3])-[CH2]-[CH2]-[CH2]-[CH](-[CH3])-[CH2]-[CH2]-[CH2]-[CH](-[CH3])-[CH2]-[CH2]-[CH2]-[CH](-[CH3])-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]",
    # aromatic
    "phenyl": "[!O;!$(C=O);!C,$([C;!H2]),$([CH2]-[C;!$(C@*)]);!$([Si](-[cH0]1[cH][cH][cH][cH][cH]1)([CH3])[CH3]);!$([Si](-[cH0]1[cH][cH][cH][cH][cH]1)(-[cH0]1[cH][cH][cH][cH][cH]1)-[$([cH0]1[cH][cH][cH][cH][cH]1),$(C([CH3])([CH3])[CH3])])]-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1", # excludes benzyl and phenoxy
    "phenoxy": "[*]-O-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1",
    "benzyl": "[!C,$([C;!X4,R]);!#1;!O]-[CH2]-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1", # excludes benzoxy
    "benzoxy": "[*]-O-[CH2]-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1",
    "benzoyl": "[*]-C(=O)-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1",
    "o-tolyl": "[*]-[cH0]1:[cH0](-[CH3]):[cH]:[cH]:[cH]:[cH]:1",
    "m-tolyl": "[*]-[cH0]1:[cH]:[cH0](-[CH3]):[cH]:[cH]:[cH]:1",
    "p-tolyl": "[!$([$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])])]-[cH0]1:[cH]:[cH]:[cH0](-[CH3]):[cH]:[cH]:1", # excludes tosyl
    "vanillyl": "[C;!$(C=O)]-[cH0]1:[cH]:[cH0](-O-[CH3]):[cH0](-[OH]):[cH]:[cH]:1",
    "vanilloyl": "[*]-C(=O)-[cH0]1:[cH]:[cH0](-O-[CH3]):[cH0](-[OH]):[cH]:[cH]:1",
    "galloyl": "[*]-C(=O)-[cH0]1:[cH]:[cH0](-O):[cH0](-O):[cH0](-O):[cH]:1",
    "mesityl": "[*]-[cH0]1:[cH0](-[CH3]):[cH]:[cH0](-[CH3]):[cH]:[cH0](-[CH3]):1",
    "2,6-diisopropylphenyl": "[*]-[cH0]1:[cH0](-[CH](-[CH3])-[CH3]):[cH]:[cH]:[cH]:[cH0](-[CH](-[CH3])-[CH3]):1",
    "2,4,6-triisopropylphenyl": "[*]-[cH0]1:[cH0](-[CH](-[CH3])-[CH3]):[cH]:[cH0](-[CH](-[CH3])-[CH3]):[cH]:[cH0](-[CH](-[CH3])-[CH3]):1",
    "2,6-di-tert-butylphenyl": "[*]-[cH0]1:[cH0](-C(-[CH3])(-[CH3])-[CH3]):[cH]:[cH]:[cH]:[cH0](-C(-[CH3])(-[CH3])-[CH3]):1",
    "2,4,6-tri-tert-butylphenyl": "[*]-[cH0]1:[cH0](-C(-[CH3])(-[CH3])-[CH3]):[cH]:[cH0](-C(-[CH3])(-[CH3])-[CH3]):[cH]:[cH0](-C(-[CH3])(-[CH3])-[CH3]):1",
    "picryl": "[*]-[cH0]1:[cH0](-[N+](=O)-[O-]):[cH]:[cH0](-[N+](=O)-[O-]):[cH]:[cH0](-[N+](=O)-[O-]):1",
    "trityl": "[*]-[CX4H0](-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1)(-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1)(-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1)",
}

MAIN_GROUP: Dict[str, str] = {
    # Organometallics
    "organo_lithium": "[#6]-[Li]",
    "Gringard": "[#6]-[Mg]-[Cl,Br,IX1]",
    "organo_mercury": "[#6]-[Hg]",
    "organo_zinc": "[#6]-[Zn]",
    "stannane": "[#6]-[SnX4](-[#6])(-[#6])-[#6]",
    "triorganotin_hydride": "[#1]-[SnX4](-[#6])(-[#6])-[#6]",
    "triorganotin_halide": "[F,Cl,Br,IX1]-[SnX4](-[#6])(-[#6])-[#1]",
    "tributyl_stannyl": "[*]-[SnX4](-[CH2]-[CH2]-[CH2]-[CH3])(-[CH2]-[CH2]-[CH2]-[CH3])-[CH2]-[CH2]-[CH2]-[CH3]",
    
    # Boron
    ## B-B
    "diboron(4)": "[BX3]-[BX3]",
    ## B-C, B-H
    ### trivalent
    "mono_organo_borane": "[#6]-[BX3;H2]",
    "di_organo_borane": "[#6]-[BX3;H1]-[#6]",
    "tri_organo_borane": "[#6]-[BX3;H0](-[#6])-[#6]",
    "9-BBN": "B1-[CH]2-[CH2][CH2][CH2]-[CH]-1-[CH2][CH2][CH2]-2",
    ### tetravalent
    "trihydro_borate": "[!#1]-[BX4-;H3]",
    "dihydro_borate": "[!#1]-[BX4-;H2]-[!#1]",
    "monohydro_borate": "[!#1]-[BX4-;H1](-[!#1])-[!#1]",
    "borate": "[BX4-;H0;!$([BX4-]1-[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-1)]", # excludes borolate
    "borolate": "[BX4-]1-[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-1",
    ## B-O
    "borinic_acid": "[#6,#1]-[BX3](-[OH,OH2+])-[#6,#1]",
    "borinic_ester": "[#6,#1]-[BX3](-O-[#6,#14])-[#6,#1]",
    "boronic_acid": "[#6,#1]-[BX3](-[OH,OH2+])-[OH,OH2+]",
    "boronic_mono_ester": "[#6,#1]-[BX3](-[O-,OH,OH2+])-O-[#6,#14]",
    "boronic_di_ester": "[#6,#1]-[BX3](-O-[#6,#14])-O-[#6,#14]",
    "catecholborane": "[*]-[BX3]1-O-c2ccccc2-O-1",
    "pinnacolborane": "[*]-[BX3]1-O-C([CH3])([CH3])-C1([CH3])[CH3]",
    "orthoborate_mono_ester": "[#6,#14,#1]-O-[BX3](-[O-,OH,OH2+])-[O-,OH,OH2+]",
    "orthoborate_di_ester": "[#6,#14,#1]-O-[BX3](-[O-,OH,OH2+])-O-[#6,#14,#1]",
    "orthoborate_tri_ester": "[#6,#14,#1]-O-[BX3](-O-[#6,#14,#1])-O-[#6,#14,#1]",
    "boroxine": "B1-O-B-O-B-O1",
    ## B-N
    "BODIPY": "[BX4-]1-[#7;X3]2~[#6;X3]~[#6;X3]~[#6;X3]~[#6;X3]~2~[#6;X3]~[#6;X3]3~[#6;X3]~[#6;X3]~[#6;X3]~[#7;X3]~3-1",
    "borazine": "[bX3-]1[nX3+][bX3-][nX3+][bX3-][nX3+]1",
    "carborazine": "[bX3-]1[nX3+]c[nX3+][bX3-]c1",
    "1,2-azaborine": "[bX3-]1[nX3+]cccc1",
    "1,3-azaborine": "[bX3-]1c[nX3+]ccc1",
    "1,4-azaborine": "[bX3-]1cc[nX3+]cc1",
    ## B-F
    "trifluoroborate": "[#6]-[BX4-](F)(F)F",

    # Silicon
    ## Si-H
    "trihydrosilane": "[#6]-[SiX4;H3]",
    "dihydrosilane": "[#6]-[SiX4;H2]-[#6]",
    "monohydrosilane": "[#6]-[SiX4;H1](-[#6])-[#6]",
    ## Si-C
    "silane": "[#6]-[SiX4](-[#6])(-[#6])-[#6]",
    "trimethylsilyl": "[*]-[SiX4;!$([Si]-O)](-[CH3])(-[CH3])-[CH3]", 
    "trimethylsilyloxy": "[*]-O-[SiX4](-[CH3])(-[CH3])-[CH3]",# excludes trimethylsilyloxy
    "triethylsilyl": "[*]-[SiX4](-[CH2]-[CH3])(-[CH2]-[CH3])-[CH2]-[CH3]",
    "dimethylisopropylsilyl": "[*]-[SiX4](-[CH3])(-[CH3])-[CH](-[CH3])-[CH3]",
    "t-butyldimethylsilyl": "[*]-[SiX4](-[CH3])(-[CH3])-C(-[CH3])(-[CH3])-[CH3]",
    "dimethylphenylsilyl": "[*]-[SiX4](-[CH3])(-[CH3])-[cH0]1[cH][cH][cH][cH][cH]1",
    "t-butyldiphenylsilyl": "[*]-[SiX4](-[cH0]1[cH][cH][cH][cH][cH]1)(-[cH0]1[cH][cH][cH][cH][cH]1)-C(-[CH3])(-[CH3])-[CH3]",
    "triisopropylsilyl": "[*]-[Si](-[CH](-[CH3])-[CH3])(-[CH](-[CH3])-[CH3])-[CH](-[CH3])-[CH3]",
    "triphenylsilyl": "[*]-[SiX4](-[cH0]1[cH][cH][cH][cH][cH]1)(-[cH0]1[cH][cH][cH][cH][cH]1)-[cH0]1[cH][cH][cH][cH][cH]1",
    ## Si-O
    "silyl_ether": "[#6;!$(C=O);!$(C=C)]-O-[SiX4](-[#6,#1])(-[#6,#1])-[#6,#1]", # excludes silyl ester, silyl enol ethers
    "silyl_ester": "[#6,#1]-C(=O)-O-[SiX4](-[#6,#1])(-[#6,#1])-[#6,#1]",
    "silyl_enol_ether": "C=C-O-[SiX4](-[#6,#1])(-[#6,#1])-[#6,#1]",
    "silanol": "[O-,OH,OH2+]-[SiX4]",
    "siloxane": "[SiX4]-O-[SiX4]",
    "dioxysilane": "[!#14]-O-[SiX4](-[#6,#1])(-[#6,#1])-O-[!#14]",
    "trioxysilane": "[!#14]-O-[SiX4](-[#6,#1])(-O-[!#14])-O-[!#14]",
    "orthosilicate": "[!#14]-O-[SiX4](-O-[!#14])(-O-[!#14])-O-[!#14]",
    # etc.
    "halosilane": "[SiX4]-[F,Cl,Br,IX1]",
    "silazane": "[SiX4]-[#7]",

    # Phosphorous
    ## P(III), P(II)
    ### PR3, PR4+
    "phosphine": "[PX3](-[#6,#14,#1])(-[#6,#14,#1])-[#6,#14,#1]",
    "phosphole": "[pX3]1:c:c:c:c:1",
    "phosphinine": "[pX2]1:c:c:c:c:c:1",
    "phosphinine_oxide": "[O-]-[pX3+]1:c:c:c:c:c:1",
    "phosphonium": "[PX4+;!$([P+]-[O-]);!$([P+]-[C-])]",
    "diphosphine": "[#6,#1]-[PX3](-[#6,#1])-[PX3](-[#6,#1])-[#6,#1]",
    "phosphaalkyne": "[#6]-[CX2]#[PX1]",
    ### PR2X
    "phosphinite": "[PX3](-[#6,#1])(-[#6,#1])-O-[#6,#14]",
    "thiophosphinite": "[PX3](-[#6,#1])(-[#6,#1])-[SX2]-[#6]",
    "aminophosphine": "[PX3](-[#6,#1])(-[#6,#1])-[#7X3,$([NX2]=[CX3])]",
    "halophosphine": "[PX3](-[#6,#1])(-[#6,#1])-[F,Cl,Br,IX1]",
    ### PRX2
    "phosphonite": "[PX3](-[#6,#1])(-O-[#6,#14])-O-[#6,#14]",
    "thiophosphonite": "[PX3](-[#6,#1])(-[SX2]-[#6])-O-[#6,#14]",
    "dithiophosphonite": "[PX3](-[#6,#1])(-[SX2]-[#6])-[SX2]-[#6]",
    "phosphonamidite": "[PX3](-[#6,#1])(-O-[#6,#14])-[#7X3,$([NX2]=[CX3])]",
    "diaminophosphine": "[PX3](-[#6,#1])(-[#7X3,$([NX2]=[CX3])])-[#7X3,$([NX2]=[CX3])]",
    "dihalophosphine": "[PX3](-[#6,#1])(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
    ### PX3
    "phosphite": "[PX3](-O-[#6,#14])(-O-[#6,#14])-O-[#6,#14]",
    "thiophosphite": "[PX3](-O-[#6,#14])(-O-[#6,#14])-[SX2]-[#6]",
    "dithiophosphite": "[PX3](-O-[#6,#14])(-[SX2]-[#6])-[SX2]-[#6]",
    "trithiophosphite": "[PX3](-[SX2]-[#6])(-[SX2]-[#6])-[SX2]-[#6]",
    "phosphoramidite": "[PX3](-O-[#6,#14])(-O-[#6,#14])-[#7X3,$([NX2]=[CX3])]",
    "phosphorodiamidite": "[PX3](-O-[#6,#14])(-[#7X3,$([NX2]=[CX3])])-[#7X3,$([NX2]=[CX3])]",
    "triaminophosphine": "[PX3](-[#7X3,$([NX2]=[CX3])])(-[#7X3,$([NX2]=[CX3])])-[#7X3,$([NX2]=[CX3])]",
    "thiophosphoramidite": "[PX3](-O-[#6,#14])(-[SX2]-[#6])-[#7X3,$([NX2]=[CX3])]",
    "dithiophosphoramidite": "[PX3](-[SX2]-[#6])(-[SX2]-[#6])-[#7X3,$([NX2]=[CX3])]",
    ## C=P(VI)
    "phosphonium_ylide": "[$([PX4+]-[C-;X3,X2]),$([PX4]=[C;X3,X2])]",
    "carbodiphosphorane": "[$([PX4]=C=[PX4]),$([PX4]=[CX2-]-[PX4+]),$([PX4+]-[CX2-2]-[PX4+])]",
    ## O=P(VI)
    ### O=PR3
    "phosphine_oxide": "[$([PX4]=O),$([PX4+]-[O-])](-[#6,#1])(-[#6,#1])-[#6,#1]",
    ### O=PR2X
    "phosphinic_acid": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-[#6])-[OH,O-,OH2+]",
    "phosphinate_ester": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-[#6])-O-[#6,#14]",
    "thiophosphinate_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-[#6])-[SX2]-[#6]",
    "phosphinamide": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-[#6])-[#7,X3](-[#6,#1])-[#6,#1]",
    "phosphinic_halide": "[$([PX4]=O),$([PX4+]-[O-])](-[#6,#1])(-[#6,#1])-[F,Cl,Br,IX1]",
    ### O=PRX2
    "phosphonic_acid": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-[OH,O-,OH2+])-[OH,O-,OH2+]",
    "phosphonate_mono_ester": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-[OH,O-,OH2+])-O-[#6,#14]",
    "phosphonate_di_ester": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-O-[#6,#14])-O-[#6,#14]",
    "thiophosphonate_ester_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-O-[#6,#14])-[SX2]-[#6]",
    "dithiophosphonate_di_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-[SX2]-[#6])-[SX2]-[#6]",
    "phosphonamidate": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-O-[#6,#14])-[#7X3,$([NX2]=[CX3])]",
    "thiophosphonamidate_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-[SX2]-[#6])-[#7X3,$([NX2]=[CX3])]",
    "phosphondiamidate": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-[#7X3,$([NX2]=[CX3])])-[#7X3,$([NX2]=[CX3])]",
    "phosphonic_dihalide": "[$([PX4]=O),$([PX4+]-[O-])](-[#6,#1])(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
    ### O=PX3
    "phosphate_mono_ester": "[$([PX4]=O),$([PX4+]-[O-])](-[OH,O-,OH2+])(-[OH,O-,OH2+])-O-[#6,#14]",
    "phosphate_di_ester": "[$([PX4]=O),$([PX4+]-[O-])](-[OH,O-,OH2+])(-O-[#6,#14])-O-[#6,#14]",
    "phosphate_tri_ester": "[$([PX4]=O),$([PX4+]-[O-])](-O-[#6,#14])(-O-[#6,#14])-O-[#6,#14]",
    "phosphate_tri_ester": "[$([PX4]=O),$([PX4+]-[O-])](-O-[#6,#14])(-O-[#6,#14])-O-[#6,#14]",
    "thiophosphate_di_ester": "[$([PX4](=O)(-[SX2H,SX1-])),$([PX4](-[OH,O-,OH2+])(=S)),$([PX4+](-[OH,O-,OH2+])(-[SX2H,SX1-]));!$([PX4+](-[SX2H])(-[OH,O-,OH2+]))](-O-[#6,#14])-O-[#6,#14]",
    "thiophosphate_mono_ester_mono_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-[OH,O-,OH2+])(-O-[#6,#14])-[SX2]-[#6]",
    "thiophosphate_di_ester_mono_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-O-[#6,#14])(-O-[#6,#14])-[SX2]-[#6]",
    "dithiophosphate_mono_ester_di_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-O-[#6,#14])(-[SX2]-[#6])-[SX2]-[#6]",
    "trithiophosphate_tri_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-[SX2]-[#6])(-[SX2]-[#6])-[SX2]-[#6]",
    "phosphoramidate_mono_ester": "[$([PX4]=O),$([PX4+]-[O-])](-[OH,O-,OH2+])(-O-[#6,#14])-[#7X3,$([NX2]=[CX3])]",
    "phosphoramidate_di_ester": "[$([PX4]=O),$([PX4+]-[O-])](-O-[#6,#14])(-O-[#6,#14])-[#7X3,$([NX2]=[CX3])]",
    "phosphorodiamidate": "[$([PX4]=O),$([PX4+]-[O-])](-O-[#6,#14])(-[#7X3,$([NX2]=[CX3])])-[#7X3,$([NX2]=[CX3])]",
    "phosphoramide": "[$([PX4]=O),$([PX4+]-[O-])](-[#7X3,$([NX2]=[CX3])])(-[#7X3,$([NX2]=[CX3])])-[#7X3,$([NX2]=[CX3])]",
    "diphosphate": "[$([PX4]=O),$([PX4+]-[O-])](-[O-,OH,$(O-[#6])])(-[O-,OH,$(O-[#6])])-O-P(-[O-,OH,$(O-[#6])])(-[O-,OH,$(O-[#6])])=O",
    "triphosphate": "[$([PX4]=O),$([PX4+]-[O-])](-[O-,OH,$(O-[#6])])(-[O-,OH,$(O-[#6])])-O-P(=O)(-[O-,OH,$(O-[#6])])-O-P(-[O-,OH,$(O-[#6])])(-[O-,OH,$(O-[#6])])=O",
    ## S=P(VI)
    ### S=PR3
    "phoshphine_sulfide": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6,#1])(-[#6,#1])-[#6,#1]",
    ### S=PR2X
    "thiophosphinate_ester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-[#6])-O-[#6,#14]",
    "dithiophosphinate_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-[#6])-[SX2]-[#6]",
    "thiophosphinamide": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-[#6])-[#7X3,$([NX2]=[CX3])]",
    "thiophosphinate_halide": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-[#6])-[F,Cl,Br,IX1]",
    ### S=PRX2
    "thiophosphonate_di_ester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-O-[#6,#14])-O-[#6,#14]",
    "dithiophosphonate_mono_ester_mono_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-O-[#6,#14])-[SX2]-[#6]",
    "trithiophosphonate_di_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-[SX2]-[#6])-[SX2]-[#6]",
    "thiophosphonamidate": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-O-[#6,#14])-[#7X3,$([NX2]=[CX3])]",
    "thiophosphondiamidate": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-[#7X3,$([NX2]=[CX3])])-[#7X3,$([NX2]=[CX3])]",
    "dithiophosphonamidate": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-[SX2]-[#6])-[#7X3,$([NX2]=[CX3])]",
    "thiophosphonate_dihalide": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
    ### S=PX3
    "thiophosphate_tri_ester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-O-[#6,#14])(-O-[#6,#14])-O-[#6,#14]",
    "dithiophosphate_di_ester_mono_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-O-[#6,#14])(-O-[#6,#14])-[SX2]-[#6]",
    "trithiophosphate_mono_ester_di_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-O-[#6,#14])(-[SX2]-[#6])-[SX2]-[#6]",
    "tetrathiophosphate_tri_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[SX2]-[#6])(-[SX2]-[#6])-[SX2]-[#6]",
    "thiophosphoramidate_mono_ester": "[$([PX4](=O)(-[SX2H,SX1-])),$([PX4](-[OH,O-,OH2+])(=S)),$([PX4+](-[OH,O-,OH2+])(-[SX2H,SX1-]));!$([PX4+](-[SX2H])(-[OH,O-,OH2+]))](-O-[#6,#14])-[#7X3,$([NX2]=[CX3])]",
    "thiophosphoramidate_di_ester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-O-[#6,#14])(-O-[#6,#14])-[#7X3,$([NX2]=[CX3])]",
    "dithiophosphoramidate_mono_ester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[SX2H,SX2-])(-O-[#6,#14])-[#7X3,$([NX2]=[CX3])]",
    "dithiophosphoramidate_mono_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[OH,O-,OH2+])(-[SX2]-[#6])-[#7X3,$([NX2]=[CX3])]",
    "dithiophosphoramidate_mono_ester_mono_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-O-[#6,#14])(-[SX2]-[#6])-[#7X3,$([NX2]=[CX3])]",
    "trithiophosphoramidate_mono_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[SX2H,SX2-])(-[SX2]-[#6])-[#7X3,$([NX2]=[CX3])]",
    "trithiophosphoramidate_di_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[SX2]-[#6])(-[SX2]-[#6])-[#7X3,$([NX2]=[CX3])]",
    "thiophosphorodiamidate_ester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-O-[#6,#14])(-[#7X3,$([NX2]=[CX3])])-[#7X3,$([NX2]=[CX3])]",
    "dithiophosphorodiamidate_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[SX2]-[#6])(-[#7X3,$([NX2]=[CX3])])-[#7X3,$([NX2]=[CX3])]",
    "thiophosphoramide": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#7X3,$([NX2]=[CX3])])(-[#7X3,$([NX2]=[CX3])])-[#7X3,$([NX2]=[CX3])]",
    "Lawesson's_reagent": "[#6]-[$([PX4]=[SX1]),$([PX4+]-[SX1-])]1-[SX2]-[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[SX2]-1)-[#6]",
    ## RN=P(VI)
    ### RN=PR3
    "iminophosphorane": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-[#6,#1])(-[#6,#1])-[#6,#1]",
    ### RN=PR2X
    "iminophosphinate": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-[#6,#1])(-[#6,#1])-O-[#6,#14]",
    "imino-thiophosphinate": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-[#6,#1])(-[#6,#1])-[SX2]-[#6,#14]",
    "iminophosphinamide": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-[#6,#1])(-[#6,#1])-[#7X3,$([NX2]=[CX3])]",
    "imidophosphoryl_monohalide": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-[#6,#1])(-[#6,#1])-[F,Cl,Br,IX1]",
    ### RN=PRX2
    "iminophosphonate": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-[#6,#1])(-O-[#6,#14])-O-[#6,#14]",
    "iminophosphonamidate": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-[#6,#1])(-O-[#6,#14])-[#7X3,$([NX2]=[CX3])]",
    "iminophosphondiamidate": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-[#6,#1])(-[#7X3,$([NX2]=[CX3])])-[#7X3,$([NX2]=[CX3])]",
    "imino-thiophosphonate_mono_ester_mono_thioester": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-[#6,#1])(-O-[#6,#14])-[SX2]-[#6,#14]",
    "imino-dithiophosphonate_di_thioester": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-[#6,#1])(-[SX2]-[#6,#14])-[SX2]-[#6,#14]",
    "imino-thiophosphonamidate_thioester": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-[#6,#1])(-[SX2]-[#6,#14])-[#7X3,$([NX2]=[CX3])]",
    "imidophosphoryl_dihalide": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-[#6,#1])(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
    ### RN=PX3
    "iminophosphate_tri_ester": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-O-[#6,#14])(-O-[#6,#14])-O-[#6,#14]",
    "iminophosphoramidate_mono_ester": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-[OH,O-,OH2+])(-O-[#6,#14])-[#7X3,$([NX2]=[CX3])]",
    "iminophosphoramidate_di_ester": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-O-[#6,#14])(-O-[#6,#14])-[#7X3,$([NX2]=[CX3])]",
    "imino-thiophosphate_di_ester_mono_thioester": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-O-[#6,#14])(-O-[#6,#14])-[SX2]-[#6,#14]",
    "imino-dithiophosphate_mono_ester_di_thioester": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-O-[#6,#14])(-[SX2]-[#6,#14])-[SX2]-[#6,#14]",
    "imino-trithiophosphate_tri_thioester": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-[SX2]-[#6,#14])(-[SX2]-[#6,#14])-[SX2]-[#6,#14]",
    "iminophosphoramidate_di_ester": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-O-[#6,#14])(-O-[#6,#14])-[#7X3,$([NX2]=[CX3])]",
    "imino-thiophosphoramidate_mono_ester_mono_thioester": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-O-[#6,#14])(-[SX2]-[#6,#14])-[#7X3,$([NX2]=[CX3])]",
    "imino-dithiophosphoramidate_di_thioester": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-[SX2]-[#6,#14])(-[SX2]-[#6,#14])-[#7X3,$([NX2]=[CX3])]",
    "iminophosphorodiamidate": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-O-[#6,#14])(-[#7X3,$([NX2]=[CX3])])-[#7X3,$([NX2]=[CX3])]",
    "iminophosphoramide": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-[#7X3,$([NX2]=[CX3])])(-[#7X3,$([NX2]=[CX3])])-[#7X3,$([NX2]=[CX3])]",
    "imidophosphoryl_trihalide": "[$([PX4]=[NX2]-[#6,#14,#1]),$([PX4+]-[NX2-]-[#6,#14,#1])](-[F,Cl,Br,IX1])(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
    "cyclotriphosphazene": "N1=[PX4]-N=[PX4]-N=[PX4]-1",
    ## PR5
    "λ5-phosphane": "[PX5]",
    
    # Sulfur
    ## S-C, S=C
    "aliphatic_thiol": "[CX4]-[SX2H]",
    "aromatic_thiol": "[cX3]-[SX2H]",
    "thioether": "[#6;!$(C=[O,S,N]);!$(C(-[O,S,N,n])-[O,S,N,n]);!$(C#N)]-[SX2;!r3;!$(S1-[#6;X3]~[#6;X3]~[O,S,N]~[#6;X3]~[#6;X3]-1)]-[#6;!$(C=[O,S,N]);!$(C([O,S,N,n])-[O,S,N,n]);!$(C#N)]", # excludes episulfides, thioesters, thioacetals, thioxines, thiocyanates
    "O,S-acetal": "[SX2H0]-[CX4;!$(C(S)(O)O)]-[OX2H0]", 
    "thioacetal": "[SX2H0]-[CX4;!$(C(S)(S)O)]-[SX2H0]",
    "thioaminal": "[SX2H0]-[CX4;!$(C(S)(N)N)]-N", 
    "ketene_O,S-acetal": "C=C(-[SX2H0])-[OX2H0]",
    "ketene_thioacetal": "C=C(-[SX2H0])-[SX2H0]",
    "ketene_thioaminal": "C=C(-[SX2H0])-N",
    "thioaldehyde": "[#6]-[CH](=[SX1])",
    "thioketone": "[#6]-C(=[SX1])-[#6]",
    "thioketene": "C=C=[SX1]",
    "thionium": "[CX3]=[SX2+]-[!$([#6;-])]",
    "carboxylate_thioester":  "[#6,#1]-C(=O)-[SX2]-[!$(C=[O,S])]",
    "thionoester": "[#6,#1]-C(=[SX1])-O-[!$(C=[O,S])]",
    "dithioester": "[#6,#1]-C(=[SX1])-[SX2]-[!$(C=[O,S])]",
    "thioamide": "[#6,#1]-C(=[SX1])-[#7X3,$([NX2]=[CX3]);!$(N-N)]",
    "thiohydrazide": "[!#7]-C(=[SX1])-[#7X3,$([NX2]=[CX3])](-N)",
    "thiohydroxamic_acid": "[#6,#1]-C(=[SX1])-[NX3]-[OH]",
    "thiohydroxamate_ester": "[#6,#1]-C(=[SX1])-[NX3]-O-[#6,#14;!$(C=O)]",
    "O-acyl_thiohydroxamate": "[#6,#1]-C(=[SX1])-[NX3]-O-C(=O)",
    "thiourea": "[#7X3,$([NX2]=[CX3])]-C(=[SX1])-[#7X3,$([NX2]=[CX3])]",
    "isothiourea": "[NX2,NX3H+]=C(-[SX2]-[#6,#14,#1])-[#7X3,$([NX2]=[CX3])]",
    "thiocarbamate": "[#7;X3,X2]-C(=O)-[SX2]-[#6,#14]",
    "thionocarbamate": "[#7;X3,X2]-C(=[SX1])-O-[#6,#14]",
    "dithiocarbamate": "[#7;X3,X2]-C(=[SX1])-[SX2]-[#6,#14]",
    "thiocarbonate": "O-C(=O)-[SX2]-[#6,#14,#1]",
    "thionocarbonate": "O-C(=[SX1])-O-[#6,#14,#1]",
    "dithiocarbonate": "[#6,#14,#1]-[SX2]-C(=O)-O-[#6,#14,#1]",
    "xanthate": "[#6,#14,#1]-O-C(=[SX1])-[SX2]-[#6,#14,#1]",
    "trithiocarbonate": "[#6,#14,#1]-[SX2]-C(=[SX1])-[SX2]-[#6,#14,#1]",
    "imidothioate": "[#6,#1]-C(=[NX2,NX3H+;!$([NX2+]=[NX1-])])-[SX2]-[#6,#14]",
    "thioimidocarbonate": "[#6,#14]-[SX2]-C(=[NX2,NX3H+;!$([NX2+]=[NX1-])])-O-[#6,#14]",
    "dithioimidocarbonate": "[#6,#14]-[SX2]-C(=[NX2,NX3H+;!$([NX2+]=[NX1-])])-[SX2]-[#6,#14]",
    "thio_anhydride": "[#6,#1]-C(=O)-[SX2]-C(=O)-[#6,#1]",
    "thiono_anhydride": "[#6,#1]-C(=[SX1])-O-C(=O)-[#6,#1]",
    "dithiono_anhydride":  "[#6,#1]-C(=[SX1])-O-C(=[SX1])-[#6,#1]",
    "trithio_anhydride": "[#6,#1]-C(=[SX1])-[SX2]-C(=[SX1])-[#6,#1]",
    "sulfonium": "[!-]-[SX3+](-[!-])-[!-]",
    "monothio_squarate": "O=c1c(=O)c(O)c1[SX2]",
    "dithio_squarate": "O=c1c(=O)c([SX2])c1[SX2]",
    ## S=C ylides
    "sulfonium_ylide": "[$([SX3+]-[C-]),$([SX3]=C)](-[#6])-[#6]",
    "sulfoxonium_ylide": "[$([SX4+](=O)-[CX3-]),$([SX4+2](-[O-])-[CX3-]),$([SX4](=O)=[CX3]),$([SX4+](-[O-])=[CX3])](-[#6])-[#6]",
    "thiocarbonyl_ylide": "[$([SX2+](=[CX3])-[CX3-]),$([SX2](-[CX3+])-[CX3-]),$([SX2](=[CX2])(=[CX2]))]",
    ## S-S, S-O, S-N
    ### divalent
    "disulfide": "[!S]-[SX2]-[SX2]-[!S]",
    "trisulfide": "[SX2]-[SX2]-[SX2]",
    "sulfenic_acid": "[#6]-[SX2]-[OH,O-,OH2+]",
    "sulfenate_ester": "[#6]-[SX2]-O-[#6]",
    "sulfenyl_halide": "[#6]-[SX2]-[F,Cl,Br,IX1]",
    "sulfoxylate_mono_ester": "[#6]-O-[SX2]-[O-,OH,OH2+]",
    "sulfoxylate_di_ester": "[#6]-O-[SX2]-O-[#6]",
    "dioxy_disulfide": "[#6]-O-[SX2]-[SX2]-O-[#6]",
    "sulfenamide": "[#6]-[SX2]-[NX3]",
    "thioxime": "[#6,#1]-C(=[NX2,NX3H+]-[SX2]-[#6,#14,#1])-[#6,#1]",
    "thiocyanate": "[*]-[SX2]-C#N",
    "isothiocyanate": "[*]-[NX2]=C=[SX1]",
    "thiocarbonyl_S-oxide": "C=[$([SX2]=[OX1]),$([SX2+]-[O-])]",
    "thiocarbonyl_S-imide": "C=[$([SX2]=[OX1]),$([SX2+]-[O-])]",
    "sulfinylamine": "[$([NX2]=[SX2]=O),$([NX2]=[SX2+]-[O-]),$([NX2-]-[SX2+2]=O),$([NX2-]-[SX2+]-[O-])]",
    "sulfur_diimide": "[$([NX2]=[SX2]=[NX2]),$([NX2]=[SX2+]-[NX2-]),$([NX2-]-[SX2+2]-[NX2-])]",
    ### trivalent
    "thiocarbonyl_S,S-dioxide": "C=[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]",
    "sulfoxide": "[#6]-[$([SX3]=O),$([SX3+]-[O-])]-[#6]",
    "thiothionyl": "[SX1]=[SX3]",
    "disulfoxide": "[#6]-[$([SX3]=O),$([SX3+]-[O-])]-[$([SX3]=O),$([SX3+]-[O-])]-[#6]",
    "sulfinic_acid": "[#6]-[$([SX3]=O),$([SX3+]-[O-])]-[OH,O-,OH2+]",
    "sulfinate_ester": "[#6]-[$([SX3]=O),$([SX3+]-[O-])]-O-[#6,#14]",
    "sulfinyl_halide": "[#6]-[$([SX3]=O),$([SX3+]-[O-])]-[F,Cl,Br,IX1]",
    "sulfinamide": "[#6]-[$([SX3]=O),$([SX3+]-[O-])]-[#7X3]",
    "N-sulfinyl_imine": "[#6]-[$([SX3]=O),$([SX3+]-[O-])]-[NX2]=C",
    "sulfite_mono_ester": "[#6]-O-[$([SX3]=O),$([SX3+]-[O-])]-[O-,OH,OH2+]",
    "sulfite_di_ester": "[#6]-O-[$([SX3]=O),$([SX3+]-[O-])]-O-[#6]",
    "amidosulfite": "[#6]-O-[$([SX3]=O),$([SX3+]-[O-])]-[#7X3,$([NX2]=[CX3])]",
    "sulfurous_diamide": "[#7X3]-[$([SX3]=O),$([SX3+]-[O-])]-[#7X3]",
    "halosulfite": "[F,Cl,Br,IX1]-[$([SX3]=O),$([SX3+]-[O-])]-O-[#6]",
    "sulfinamide": "[#6]-[$([SX3]=O),$([SX3+]-[O-])]-[#7X3]",
    "amidosulfite": "[#6]-O-[$([SX3]=O),$([SX3+]-[O-])]-O-[#6]",
    "thiosulfite": "[#6]-O-[$([SX3]=O),$([SX3+]-[O-])]-[SX2]-[#6]",
    "thionosulfite": "[#6]-O-[$([SX3]=[SX1]),$([SX3+]-[SX1-])]-O-[#6]",
    "thiosulfinate": "[#6]-[$([SX3]=O),$([SX3+]-[O-])]-[SX2]-[#6,#14]",
    "sulfilimine": "[#6]-[$([SX3]=[NX2]),$([SX3+]-[NX2-])]-[#6]",
    "sulfinimidate": "[#6]-[$([SX3]=[NX2]),$([SX3+]-[NX2-])]-O-[#6]",
    "sulfinamidine": "[#6]-[$([SX3]=[NX2]),$([SX3+]-[NX2-])]-[#7X3,$([NX2]=[CX3])]",
    ### tetravalent
    "λ4-sulfane": "[SX4](-[*])(-[*])(-[*])-[*]",
    "sulfone": "[#6]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[#6]",
    "sulfone": "[#6]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[#6]",
    "sulfonic_acid": "[#6]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[OH,O-,OH2+]",
    "sulfonate_ester": "[#6]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-O-[#6,#14]",
    "sulfonyl_halide": "[#6]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[F,Cl,Br,IX1]",
    "thiosulfonate": "[#6]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[SX2]-[#6,#14]",
    "sulfonamide": "[#6]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[#7X3;!$([#7]([$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])])-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])])]",
    "N-sulfonyl_imine": "[#6]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[NX2]=[CX3]",
    "sulfonimide": "[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[#7;X3]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[#6]",
    "sulfamide": "[#7X3,$([NX2]=[CX3])]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[#7X3,$([NX2]=[CX3])]",
    "sulfamic_acid": "[#7X3,$([NX2]=[CX3])]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[OH,O-,OH2+]",
    "sulfamate_ester": "[#7X3,$([NX2]=[CX3])]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-O-[#6,#14]",
    "sulfamoyl_halide": "[#7X3,$([NX2]=[CX3])]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[F,Cl,Br,IX1]",
    "sulfate_mono_ester": "[OH,O-,OH2+]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-O-[#6,#14]",
    "sulfate_di_ester": "[#6,#14]-O-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-O-[#6,#14]",
    "halosulfate": "[F,Cl,Br,IX1]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-O-[#6,#14]",
    "Bunte_salt": "[OH,O-,OH2+]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[SX2]-[#6,#14]",
    "sulfoximine": "[#6]-[$([SX4](=O)=[NX2]),$([SX4+](=O)-[NX2-]),$([SX4+](-[O-])=[NX2]),$([SX4+2](-[O-])-[NX2-])]-[#6]",
    "sulfonimidate": "[#6]-[$([SX4](=O)=[NX2]),$([SX4+](=O)-[NX2-]),$([SX4+](-[O-])=[NX2]),$([SX4+2](-[O-])-[NX2-])]-O-[#6,#14]",
    "sulfonimidoyl_halide": "[#6]-[$([SX4](=O)=[NX2]),$([SX4+](=O)-[NX2-]),$([SX4+](-[O-])=[NX2]),$([SX4+2](-[O-])-[NX2-])]-[F,Cl,Br,IX1]",
    "sulfonimidamide": "[#6]-[$([SX4](=O)=[NX2]),$([SX4+](=O)-[NX2-]),$([SX4+](-[O-])=[NX2]),$([SX4+2](-[O-])-[NX2-])]-[#7X3,$([NX2]=[CX3])]",
    "sulfondiimidoyl_halide": "[#6]-[$([SX4](=[NX2])=[NX2]),$([SX4+](=[NX2])-[NX2-]),$([SX4+2](-[NX2-])-[NX2-])]-[F,Cl,Br,IX1]",
    "imidosulfuric_diamide": "[#7X3,$([NX2]=[CX3])]-[$([SX4](=O)=[NX2]),$([SX4+](=O)-[NX2-]),$([SX4+2](-[O-])-[NX2-]),$([SX4+](-[O-])=[NX2])]-[#7X3,$([NX2]=[CX3])]",
    "trifluorosulfanyl": "[*]-[SX4](-F)(-F)-F",
    "sulfoxonium": "[#6;!-]-[$([SX4+]=O),$([SX4+2]-[O-])](-[#6;!-])-[#6;!-]", # excludes sulfoxonium ylides
    ### common sulfonyls
    "mesyl": "[*]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[CH3]",
    "tosyl": "[*]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[cH0]1:[cH]:[cH]:[cH0](-[CH3]):[cH]:[cH]:1",
    "o-nosyl": "[*]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[cH0]1:[cH0](-[N+](=O)(-[O-])):[cH]:[cH1]:[cH]:[cH]:1",
    "p-nosyl": "[*]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[cH0]1:[cH]:[cH]:[cH0](-[N+](=O)(-[O-])):[cH]:[cH]:1",
    "bresyl": "[*]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[cH0]1:[cH]:[cH]:[cH0](-Br):[cH]:[cH]:1",
    "triflyl": "[*]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-C(F)(F)F",
    "dansyl": "[*]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[cH0]1:[cH]:[cH]:[cH]:[cH0]2:[cH0](-[NX3](-[CH3])-[CH3]):[cH]:[cH]:[cH]:[cH0]:1:2", # 5-dimethylaminonaphthalene
    "nonaflyl": "[*]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)F", # C4F9
    ### hexavalent
    "λ6-sulfane": "[SX6]",

    # Halogens
    ## C(sp3)-X
    "alkyl_fluoride": "F-[CX4](-[!$([F,Cl,Br,I])])(-[!$([F,Cl,Br,I])])-[!$([F,Cl,Br,I])]",
    "alkyl_chloride":  "Cl-[CX4](-[!$([F,Cl,Br,I])])(-[!$([F,Cl,Br,I])])-[!$([F,Cl,Br,I])]",
    "alkyl_bromide":  "Br-[CX4](-[!$([F,Cl,Br,I])])(-[!$([F,Cl,Br,I])])-[!$([F,Cl,Br,I])]",
    "alkyl_iodide":  "[IX1]-[CX4](-[!$([F,Cl,Br,I])])(-[!$([F,Cl,Br,I])])-[!$([F,Cl,Br,I])]",
    ## C(sp2)-X
    "vinyl_fluoride": "C=[CX3]-F",
    "vinyl_chloride": "C=[CX3]-Cl",
    "vinyl_bromide": "C=[CX3]-Br",
    "vinyl_iodide": "C=[CX3]-[IX1]",
    "aryl_fluoride": "c-F",
    "aryl_chloride": "c-Cl",
    "aryl_bromide": "c-Br",
    "aryl_iodide": "c-[IX1]",
    "acyl_fluoride": "[!$([O,N,S])]-C(=O)-F",
    "acyl_chloride": "[!$([O,N,S])]-C(=O)-Cl",
    "acyl_bromide": "[!$([O,N,S])]-C(=O)-Br",
    "acyl_iodide": "[!$([O,N,S])]-C(=O)-[IX1]",
    "thioacyl_fluoride": "[!$([O,N,S])]-C(=[SX1])-F",
    "thioacyl_chloride": "[!$([O,N,S])]-C(=[SX1])-Cl",
    "thioacyl_bromide": "[!$([O,N,S])]-C(=[SX1])-Br",
    "thioacyl_iodide": "[!$([O,N,S])]-C(=[SX1])-[IX1]",
    "imidoyl_fluoride": "[#6,#14,#1]-C(=[NX2,NX3H+;!r3])-F",
    "imidoyl_chloride": "[#6,#14,#1]-C(=[NX2,NX3H+;!r3])-Cl",
    "imidoyl_bromide": "[#6,#14,#1]-C(=[NX2,NX3H+;!r3])-Br",
    "imidoyl_iodide": "[#6,#14,#1]-C(=[NX2,NX3H+;!r3])-[IX1]",
    "halo_formate": "[F,Cl,Br,IX1]-C(=O)-O-[#6,#14]",
    "halo_formamide": "[F,Cl,Br,IX1]-C(=O)-[#7]",
    "halo_formamidine": "[F,Cl,Br,IX1]-C(=[NX2,NX3H+]-[#6,#14,#1])-[NX3](-[#6,#14,#1])-[#6,#14,#1]",
    "halo_formimidate": "[F,Cl,Br,IX1]-C(=[NX2,NX3H+;!$([NX2+]=[NX1-])])-O-[#6,#14]",
    "halo_formamidoxime": "[F,Cl,Br,IX1]-C(=[NX2,NX3H+]-O)-[NX3]",
    "halo_formamidrazone": "[F,Cl,Br,IX1]-[CX3](-,=N-[NX3])-,=N",
    "halo_thioformate":  "[F,Cl,Br,IX1]-C(=O)-[SX2]-[#6,#14]",
    "halo_thionoformate": "[F,Cl,Br,IX1]-C(=[SX1])-O-[#6,#14]",
    "halo_dithioformate": "[F,Cl,Br,IX1]-C(=[SX1])-[SX2]-[#6,#14]",
    "halo_thioformamide": "[F,Cl,Br,IX1]-C(=[SX1])-[#7]",
    "halo_thioformimidate": "[F,Cl,Br,IX1]-C(=[NX2,NX3H+;!$([NX2+]=[NX1-])])-[SX2]-[#6,#14]",
    "phosgene_oxime": "[F,Cl,Br,IX1]-C(=[NX2,NX3H+]-O)-[F,Cl,Br,IX1]",
    "phosgene_hydrazone": "[F,Cl,Br,IX1]-C(=[NX2,NX3H+]-[#7])-[F,Cl,Br,IX1]",
    "oxalohalide": "[#6,#14]-C(=O)-C(=O)-[F,Cl,Br,IX1]",
    ## C(sp)-X
    "alkynyl_fluoride": "C#[CX2]-F",
    "alkynyl_chloride": "C#[CX2]-Cl",
    "alkynyl_bromide": "C#[CX2]-Br",
    "alkynyl_iodide": "C#[CX2]-[IX1]",
    ## 2+ X
    "geminal_dihalide": "[!$([F,Cl,Br,I])]-[CX4;!$(C(F)(F)-C(F)(F))](-[F,Cl,Br,IX1])(-[F,Cl,Br,IX1])-[!$([F,Cl,Br,I])]",
    "vicinal_dihalide": "[F,Cl,Br,IX1]-[CX4](-[!$([F,Cl,Br,I])])(-[!$([F,Cl,Br,I])])-[CX4](-[!$([F,Cl,Br,I])])(-[!$([F,Cl,Br,I])])-[F,Cl,Br,IX1]",
    "trifluoromethyl": "[!$([cH0]1[cH][cH0](-C(F)(F)F)[cH][cH0][cH]1);!$([cH0]1[cH0]c(-C(F)(F)F)[cH]c(-C(F)(F)F)[cH]1);!$([cH0]1[cH]c(-C(F)(F)F)[cH0]c(-C(F)(F)F)[cH]1);!$(C(F)(F));!$(C(=O)O);!$([$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])])]-C(F)(F)F",
    "trifluoroacetoxy": "FC(F)(F)-C(=O)-O-[!#1]",
    "trihalomethyl": "[*]-[C;!$(C(F)(F)F)](-[F,Cl,Br,IX1])(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]", # excluding trifluoromethyl
    "2,4,6-trifluorophenyl": "[*]-[cH0]1:[cH0](-F):[cH]:[cH0](-F):[cH]:[cH0](-F):1",
    "pentafluorophenyl": "[*]-c1c(F)c(F)c(F)c(F)c1(F)",
    "3,5-bis(trifluoromethyl)phenyl": "[*]-[cH0]1:[cH]:[cH0](-C(F)(F)F):[cH]:[cH0](-C(F)(F)F):[cH]:1",
    "2,4,6-tris(trifluoromethyl)phenyl": "[*]-[cH0]1:[cH0](-C(F)(F)F):[cH]:[cH0](-C(F)(F)F):[cH]:[cH0](-C(F)(F)F):1",
    ### PFAs
    "perfluoro-ethyl": "[!$(C(F)(F))]-C(F)(F)-C(F)(F)F",
    "perfluoro-propyl": "[!$(C(F)(F))]-C(F)(F)-C(F)(F)-C(F)(F)F",
    "perfluoro-isopropyl": "[*]-C(F)(-C(F)(F)F)-C(F)(F)F",
    "perfluoro-butyl": "[!$(C(F)(F))]-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)F",
    "perfluoro-pentyl": "[!$(C(F)(F))]-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)F",
    "perfluoro-hexyl": "[!$(C(F)(F))]-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)F",
    "perfluoro-heptyl": "[!$(C(F)(F))]-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)F",
    "perfluoro-octyl": "[*]-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)F",
    ## O-X, N-X
    "hypohalite": "[*]-O-[F,Cl,Br,IX1]",
    "halamine": "[!$([F,Cl,Br,I])]-[NX3](-[!$([F,Cl,Br,I])])-[F,Cl,Br,IX1]",
    "fluoroammonium": "F-[NX4+](-[#6])(-[#6])-[#6]",
    "dihaloamine": "[!$([F,Cl,Br,I])]-[NX3](-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
    ## hypervalent iodine
    ### I(III)
    "iodosyl": "[*]-[$([IX2]=O),$([IX2+]-[O-])]",
    "iminoiodinane": "[$([IX2]=[NX2]),$([IX2+]-[NX2-])]",
    "difluoro_iodane": "[*]-[IX3](-F)-F",
    "dichloro_iodane": "[*]-[IX3](-Cl)-Cl",
    "dihalo_iodane": "[*]-[IX3;!$(I(F)F);!$(I(Cl)Cl)](-[F,Cl,Br])-[F,Cl,Br]",
    "monohalo_iodane": "[!$([F,Cl,Br])]-[IX3](-[!$([F,Cl,Br])])-[F,Cl,Br]",
    "diacyloxy_iodane":  "[!$(O-C(=O))]-[IX3](-O-C(=O)-[#6])-O-C(=O)-[#6]",
    "triacyloxy_iodane":  "[#6]-C(=O)-O-[IX3](-O-C(=O)-[#6])-O-C(=O)-[#6]",
    "λ3-iodoxole": "[IX3,IX2+]1-O-[CX4]-[c;R2][c;R2,R3]-1",
    "λ3-iodoxolone": "[*]-[IX3]1-O-C(=O)-[c;R2]:[c;R2,R3]-1",
    "λ3-iodoxole_sulfone": "[*]-[IX3]1-O-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[c;R2]:[c;R2,R3]-1",
    "hydroxy_iodane": "[*]-[IX3](-[*])-[OH]",
    "sulfonyloxy_iodane": "[*]-[IX3](-[*])-O-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[#6]",
    "diaryl_iodonium": "[c]-[IX2+]-[c]",
    "aryl_vinyl_iodonium": "[c]-[IX2+]-C=C",
    "aryl_akynyl_iodonium": "[c]-[IX2+]-C#C",
    "hydroxy_iodonium": "[OH]-[IX2+]-[*]",
    "iodonium_ylide": "[*]-[$([IX2+]-[CX3-]),$([IX2]=[CX3])]",
    "iodonium": "[*]-[IX2+]-[*]",
    ### I(V)
    "iodyl": "[*]-[$([IX3](=O)=O),$([IX3+](=O)(-[O-])),$([IX3+2](-[O-])(-[O-]))]",
    "λ5-iodoxole": "O-[$([IX4]=O),$([IX4+]-[O-])]1-O-[CX4]-[c;R2]:[c;R2,R3]-1",
    "λ5-iodoxolone": "O-[$([IX4]=O),$([IX4+]-[O-])]1-O-C(=O)-[c;R2]:[c;R2,R3]-1",
    "λ5-iodoxolsulfone": "O-[$([IX4]=O),$([IX4+]-[O-])]1-O-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[c;R2]:[c;R2,R3]-1",
    "iodoxole_periodinane": "O-[IX5]1(-O)(-O)-O-[CX4]-[c;R2]:[c;R2,R3]-1",
    "iodoxolone_periodinane": "O-[IX5]1(-O)(-O)-O-C(=O)-[c;R2]:[c;R2,R3]-1",
    "iodoxolsulfone_periodinane": "O-[IX5]1(-O)(-O)-O-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[c;R2]:[c;R2,R3]-1",

    # Arsenic
    "arsine": "[#6,#14,#1]-[AsX3](-[#6,#14,#1])-[#6,#14,#1]",
    "monohalo_arsine": "[F,Cl,Br,IX1]-[AsX3](-[#6,#14,#1])-[#6,#14,#1]",
    "dihalo_arsine": "[F,Cl,Br,IX1]-[AsX3](-[#6,#14,#1])-[F,Cl,Br,IX1]",
    "diarsine": "[#6,#14,#1]-[AsX3](-[#6,#14,#1])-[AsX3](-[#6,#14,#1])-[#6,#14,#1]",
    "arsine_oxide": "[#6,#14,#1]-[$([AsX4]=O),$([AsX4+]-[O-])](-[#6,#14,#1])-[#6,#14,#1]",
    "arsonium": "[AsX4+;!$([As+]-[O-]);!$([P+]-[C-])]",
    "arsonium_ylide": "[$([AsX4+]-[C-;X3,X2]),$([AsX4]=[C;X3,X2])]",
    "arsinite": "[#6,#14,#1]-[AsX3](-[#6,#14,#1])-O",
    "thioarsinite": "[#6,#14,#1]-[AsX3](-[#6,#14,#1])-[SX2,SX1-]",
    "arsonite": "[#6,#14,#1]-[AsX3](-O)-O",
    "monothio_arsonite": "[#6,#14,#1]-[AsX3](-O)-[SX2,SX1-]",
    "dithio_arsonite": "[#6,#14,#1]-[AsX3](-[SX2,SX1-])-[SX2,SX1-]",
    "aminoarsine": "[#6,#14,#1]-[AsX3](-[#6,#14,#1])-[#7X3,$([NX2]=[CX3])]",
    "diaminoarsine": "[#6,#14,#1]-[AsX3](-[#7X3,$([NX2]=[CX3])])-[#7X3,$([NX2]=[CX3])]",
    "triaminoarsine": "[#7X3,$([NX2]=[CX3])]-[AsX3](-[#7X3,$([NX2]=[CX3])])-[#7X3,$([NX2]=[CX3])]",
    "arsinic_acid": "[#6]-[$([AsX4]=O),$([AsX4+]-[O-])](-[#6])-[O-,OH,OH2+]",
    "arsinyl_halide": "[#6]-[$([AsX4]=O),$([AsX4+]-[O-])](-[#6])-[F,Cl,Br,IX1]",
    "arsinate_ester": "[#6]-[$([AsX4]=O),$([AsX4+]-[O-])](-[#6])-O-[#6,#14]",
    "arsonic_acid": "[#6]-[$([AsX4]=O),$([AsX4+]-[O-])](-[O-,OH,OH2+])-[O-,OH,OH2+]",
    "arsenyl_dihalide": "[#6]-[$([AsX4]=O),$([AsX4+]-[O-])](-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
    "arsonate_mono_ester": "[#6]-[$([AsX4]=O),$([AsX4+]-[O-])](-[O-,OH,OH2+])-O-[#6,#14]",
    "arsonate_di_ester": "[#6]-[$([AsX4]=O),$([AsX4+]-[O-])](-O-[#6,#14])-O-[#6,#14]",
    "arsenite": "O-[AsX3](-O)-O",
    "monothio_arsenite": "O-[AsX3](-O)-[SX2,SX1-]",
    "dithio_arsenite": "O-[AsX3](-[SX2,SX1-])-[SX2,SX1-]",
    "trithio_arsenite": "[SX2,SX1-]-[AsX3](-[SX2,SX1-])-[SX2,SX1-]",
    "arsenic_acid": "[O-,OH,OH2+]-[$([AsX4]=O),$([AsX4+]-[O-])](-[O-,OH,OH2+])-[O-,OH,OH2+]",
    "arsenate_mono_ester": "[O-,OH,OH2+]-[$([AsX4]=O),$([AsX4+]-[O-])](-[O-,OH,OH2+])-O-[#6,#14]",
    "arsenate_di_ester": "[O-,OH,OH2+]-[$([AsX4]=O),$([AsX4+]-[O-])](-O-[#6,#14])-O-[#6,#14]",
    "arsenate_tri_ester": "[#6,#14]-O-[$([AsX4]=O),$([AsX4+]-[O-])](-O-[#6,#14])-O-[#6,#14]",
    "thiono_arsenate": "[$([AsX4]=[SX1]),$([AsX4+]-[SX1-])](-O)(-O)-O",
    "monothio_arsenate": "[$([AsX4]=O),$([AsX4+]-[O-])](-O)(-O)-[SX2,SX1-]",
    "dithio_arsenate": "[$([AsX4]=O),$([AsX4+]-[O-])](-O)(-[SX2,SX1-])-[SX2,SX1-]",
    "trithio_arsenate": "[$([AsX4]=O),$([AsX4+]-[O-])](-[SX2,SX1-])(-[SX2,SX1-])(-[SX2,SX1-])",
    "tetrathio_arsenate": "[$([AsX4]=[SX1]),$([AsX4+]-[SX1-])](-[SX2,SX1-])(-[SX2,SX1-])-[SX2,SX1-]",

    # Selenium
    "selenoether": "[#6;!$(C#N)]-[SeX2;!r3]-[#6;!$(C#N)]",
    "selenophene": "[#34;X2]1:c:c:c:c1",
    "diselenide": "[SeX2]-[SeX2]",
    "selenol": "[#6;!$(C=[O,S,N])]-[SeX2H]",
    "selenamide": "[*]-[SeX2]-[#7]",
    "selenoester": "[#6,#1]-C(=O)-[SeX2]-[#6,#14]",
    "selanyl_halide": "[SeX2]-[F,Cl,Br,IX1]",
    "selenoxide": "[#6]-[$([SeX3]=O),$([SeX3+]-[O-])]-[#6]",
    "selenonium": "[!-]-[SeX3+](-[!-])-[!-]",
    "selenonium_ylide": "[$([SeX3+]-[C-]),$([SeX3]=C)](-[#6])-[#6]",
    "seleninic_acid": "[#6]-[$([SeX3]=O),$([SeX3+]-[O-])]-[O-,OH,OH2+]",
    "seleninate_ester": "[#6]-[$([SeX3]=O),$([SeX3+]-[O-])]-O-[#6,#14]",
    "selenonic_acid": "[#6]-[$([SeX4](=O)=O),$([SeX4+](=O)-[O-]),$([SeX4+2](-[O-])-[O-])]-[O-,OH,OH2+]",
    "selenonate_ester": "[#6]-[$([SeX4](=O)=O),$([SeX4+](=O)-[O-]),$([SeX4+2](-[O-])-[O-])]-O-[#6,#14]",
    "selenocyanate": "[*]-[SeX2]-[CX2]#[NX1]",
    "isoselenocyanate": "[*]-[NX2]=[CX2]=[SeX1]",
    "λ4-selenane": "[SeX4](-[*])(-[*])(-[*])-[*]", 


}

# [c;!$(c(:a)(:a)(:a))] means an aromatic C that is NOT connected to 3 aromatic atoms via aromatic bonds.
# This prevents additional fused aromatic rings.

HOMOAROMATICS: Dict[str, str] = {
    # 1 ring
    ## 3-membered
    "cyclopropenium":  "[c+]1cc1",
    ## 4-membered
    "cyclobutadiene_dianion":  "[c-]1[c-]cc1",
    "cyclobutadiene": "[#6;X3;!$([C;R2]12=CC@2=C1);!$(c12ccccc2-c2ccccc21)]1=,:[#6;X3;!$([C;R2]12=CC@2=C1)]-[#6;X3;!$([C;R2]12=CC@2=C1)]=,:[#6;X3;!$([C;R2]12=CC@2=C1)]1",
    "cyclobutadiene_dication": "[c+]1[c+]cc1",
    ## 5-membered
    "cyclopentadienide": "[c-]1cccc1",
    ## 6+ membered 
    "benzenoid_ring": "[cX3]1~[cX3]~[cX3]~[cX3]~[cX3]~[cX3]~1",
    "ortho-phenylene": "[!$([CH3])]-[cH0]1:[cH0](-[!$([CH3])]):[cH]:[cH]:[cH]:[cH]:1", # excludes o-tolyl
    "meta-phenylene": "[!$([CH3])]-[cH0]1:[cH]:[cH0](-[!$([CH3])]):[cH]:[cH]:[cH]:1", # excludes m-tolyl
    "para-phenylene": "[!$([CH3])]-[cH0]1:[cH]:[cH]:[cH0](-[!$([CH3])]):[cH]:[cH]:1", # excludes p-tolyl
    "biphenyl": "[c;!$(c-!@c2ccccc2);!$(c(:a)(:a)(:a))]1:[c;!$(c-!@c2ccccc2);!$(c(:a)(:a)(:a))]:[c;!$(c-!@c2ccccc2);!$(c(:a)(:a)(:a))]:[c;!$(c-!@c2ccccc2);!$(c(:a)(:a)(:a))]:[c;!$(c-!@c2ccccc2);!$(c(:a)(:a)(:a))]:c:1-!@c1:[c;!$(c-!@c2ccccc2);!$(c(:a)(:a)(:a))]:[c;!$(c-!@c2ccccc2);!$(c(:a)(:a)(:a))]:[c;!$(c-!@c2ccccc2);!$(c(:a)(:a)(:a))]:[c;!$(c-!@c2ccccc2);!$(c(:a)(:a)(:a))]:[c;!$(c-!@c2ccccc2);!$(c(:a)(:a)(:a))]:1", # excludes terphenyls biphenylene, fluorene, etc.
    "biaryl": "[c;!$([c;!$(c(:a)(:a)(:a))]1[c;!$(c(:a)(:a)(:a))][c;!$(c(:a)(:a)(:a))][c;!$(c(:a)(:a)(:a))][c;!$(c(:a)(:a)(:a))][c;!$(c(:a)(:a)(:a))]1)]-!@c",
    "ortho-terphenyl": "c1ccccc1-!@c1c(-!@c2ccccc2)cccc1",
    "meta-terphenyl": "c1ccccc1-!@c1cc(-!@c2ccccc2)ccc1",
    "para-terphenyl": "c1ccccc1-!@c1ccc(-!@c2ccccc2)cc1",
    "benzyne": "c1#ccccc1",
    "tropylium": "[c+]1cccccc1",

    # !$(c1:c~[#6;X3]~[#6;X3]~c:1) prevents 5-membered all-sp2 rings at bridgeheads. Used to exclude acenaphthylene and fluoranthene.
    # !$([cR2;r6]) prevents fusing with 6-membered rings but allows fusing with 5-membered rings.

    # 2 rings
    "indene": "[CX4]1C=C[cX3H0]2:c:c:c:c:[cX3H0]:2-1",
    "naphthalene": "[c;!$([cR2;r6](:a)(:a)(:a))]1:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0;!$(c1:c~[#6;X3]~[#6;X3]~c:1)]2:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0;!$(c1:c~[#6;X3]~[#6;X3]~c:1)]:1:2",
    "azulene": "c1:c:c:c2:c:c:c:c:c:c:1-2",
    # 3 rings
    "anthracene": "[c;!$([cR2;r6](:a)(:a)(:a))]1:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]2:[c;!$(c(:a)(:a)(:a))]:[cX3H0]3:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]:3:[c;!$(c(:a)(:a)(:a))]:[cX3H0]:1:2",
    "phenanthrene": "[c;!$([cR2;r6](:a)(:a)(:a))]1:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]2:[cX3H0]3:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0;!$(c1:c~[#6;X3]~[#6;X3]~c:1)]:3:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0;!$(c1:c~[#6;X3]~[#6;X3]~c:1)]:1:2",   
    "biphenylene": "c1:c:c:c:[cX3H0]2-[cX3H0]3:c:c:c:c:[cX3H0]:3-[cX3H0]:1:2",
    "fluorene": "c1:c:c:c:[cX3H0]2-[cX3H0]3:c:c:c:c:[cX3H0]:3-[CX4;!$([CH]-[CH2]-O-[$(C(=O)(O)[!#6])])]-[cX3H0]:1:2",
    "acenaphthylene": "c1:c:c:[cX3H0](-[#6;X3]~[#6;X3;!$(c1ccccc1)]3):[cX3H0]2:[cX3H0]-3:c:c:c:[cX3H0]:1:2", # excludes fluoranthene
    # 4 rings
    "tetracene":  "[c;!$([cR2;r6](:a)(:a)(:a))]1:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]2:[c;!$(c(:a)(:a)(:a))]:[cX3H0]3:[c;!$(c(:a)(:a)(:a))]:[cX3H0]4:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]:4:[c;!$(c(:a)(:a)(:a))]:[cX3H0]:3:[c;!$(c(:a)(:a)(:a))]:[cX3H0]:1:2",
    "tetraphene": "[c;!$([cR2;r6](:a)(:a)(:a))]1:[c;!$([cR2;r6]c(:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]2:[c;!$(c(:a)(:a)(:a))]:[cX3H0]3:[cX3H0]4:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]:4:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]:3:[c;!$(c(:a)(:a)(:a))]:[cX3H0]:1:2",
    "chrysene":   "[c;!$([cR2;r6](:a)(:a)(:a))]1:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]2:[cX3H0]3:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]4:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]:4:[cX3H0]:3:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]:1:2",
    "benzo[c]phenanthrene": "[c;!$([cR2;r6](:a)(:a)(:a))]1:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]2:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]3:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]4:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]:4:[cX3H0]:3:[cX3H0]:1:2",
    "pyrene": "c1:c:c:[cX3H0](:[cX3H0]2:[cX3H0]34):c:c:[cX3H0]:4:c:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2",   
    "triphenylene": "c1:c:c:c:[cX3H0]2:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]4:c:c:c:c:[cX3H0]:4:[cX3H0]:1:2",   
    "fluoranthene": "c1:c:c:[cX3H0](-[cX3H0]4:c:c:c:c:[cX3H0]5:4):[cX3H0]2:[cX3H0]-5:c:c:c:[cX3H0]:1:2",
    # 5 rings
    "perylene": "[cX3H0]12:[cX3H0]3:c:c:c:[cX3H0]:1:c:c:c:[cX3H0]:2:[cX3H0]4:c:c:c:[cX3H0]5:c:c:c:[cX3H0]:3:[cX3H0]:4:5",
}

HETEROAROMATICS: Dict[str, str] = {
    # [c;!$(:a)(:a)(:a)] means an aromatic C that is NOT connected to 3 aromatic atoms. 
    # This prevents additional fused aromatic rings.

    # 5-membered rings
    ## 1 hetero atomx
    "pyrrole": "[nX3,nX2-;!$([#7]@[BX4]@[#7])]1:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1", # excludes BODIPY
    "N-amino_pyrrole": "[#7;!n]-n1cccc1",
    "N-oxy_pyrrole": "[#8]-n1cccc1",
    "any_pyrrole": "[#7;!X4]1~[#6;X3]~[#6;X3]~[#6;X3]~[#6;X3]~1", 
    "furan": "[oX2]1:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
    "any_furan": "o1:c:c:c:c:1", 
    "thiophene": "[sX2]1:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
    "thiophene_oxide": "[$([s+]-[O-]),$(S=O)]1-,:[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-1", 
    "any_thiophene": "[s;!$(s~O)]1:c:c:c:c:1", 
    
    ## 2 hetero atoms
    "pyrazole": "[nX3,nX2-]1:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
    "isoxazole": "[oX2]1:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
    "isothiazole": "[sX2]1:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
    "imidazole": "[nX3,nX2-]1:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
    "oxazole": "[oX2]1:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
    "thiazole": "[sX2]1:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
    ## 3 hetero atoms
    "1,2,3-triazole": "n1:n:n:[c;!$(c1ccccc1);!$(c1ncccc1);!$(c1cnccc1)]:[c;!$(c1ccccc1);!$(c1ncccc1);!$(c1cnccc1)]:1", # either tautomer
    "1,2,4-triazole": "n1:n:[c;!$(c(:a)(:a)(:a))]:n:[c;!$(c(:a)(:a)(:a))]:1", # either tautomer
    "1,2,3-oxadiazole": "[oX2]1:[nX2,nX3+]:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
    "1,2,4-oxadiazole": "[oX2]1:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:1",
    "1,2,5-oxadiazole": "[oX2]1:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3+]:1",
    "1,3,4-oxadiazole": "[oX2]1:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3+]:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:1",
    "1,2,3-thiadiazole": "[sX2]1:[nX2,nX3+]:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
    "1,2,4-thiadiazole": "[sX2]1:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:1",
    "1,2,5-thiadiazole": "[sX2]1:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3+]:1",
    "1,3,4-thiadiazole": "[sX2]1:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3+]:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:1",
    ## 4 hetero
    "tetrazole": "[c;!$(c(:a)(:a)(:a))]1:n:n:n:n:1", # either tautomer

    # 6-membered rings
    ## 1 hetero atom
    "pyridine": "[nX2,nX3H+]1:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
    "pyridinium": "[nX2,nX3H0+]1:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
    "any_pyridine": "[#7X2,n]1~[cX3]~[cX3]~[cX3]~[cX3]~[cX3]~1",
    "pyrylium": "[oX3+]1:c:c:c:c:c:1",
    "thiopyrylium": "[sX3+]1:c:c:c:c:c:1",  
    ## 2 hetero atoms
    "pyridazine": "[nX2,nX3+]1:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
    "pyrimidine": "[nX2,nX3+]1:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
    "pyrazine": "[nX2,nX3+]1:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
    # RDKit doesn't consider these to be aromatic (which is correct)
    "1,4-dihydropyrazine": "[N;!$(N1-[#6]~[#6]~[#6]-1)]1-[C;!r3]=[C;!r3]-[N;!$(N1-[#6]~[#6]~[#6]-1)]-[C;!r3]=[C;!r3]-1",
    "1,2-dioxine": "[O;!$(O1-O-[#6]~[#6]-1)]1-[O;!$(O1-O-[#6]~[#6]-1)]-[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-1", # allowed to have fused aromatics
    "1,4-dioxine": "[O;!$(O1-[#6]~[#6]~[#6]-1)]1-C=C-[O;!$(O1-[#6]~[#6]~[#6]-1)]-C=C-1", # not allowed
    "1,2-dithiine": "[S;!$(S1-S-[#6]~[#6]-1)]1-[S;!$(S1-S-[#6]~[#6]-1)]-[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-1", # allowed to have fused aromatics
    "1,4-dithiine": "[S;!$(S1-[#6]~[#6]~[#6]-1)]1-C=C-[S;!$(S1-[#6]~[#6]~[#6]-1)]-C=C-1", # not allowed
    "1,2-oxathiine": "[O;!$(O1-S-[#6]~[#6]-1)]1-[S;!$(S1-O-[#6]~[#6]-1)]-[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-1", # allowed to have fused aromatics
    "1,4-oxathiine": "[O;!$(O1-[#6]~[#6]~[#6]-1)]1-C=C-[S;!$(S1-[#6]~[#6]~[#6]-1)]-C=C-1", # not allowed
    "1,2-oxazine": "[O;!$(O1-N-[#6]~[#6]-1)]1-[N;!$(N1-O-[#6]~[#6]-1)]-C=C-C=C-1", # not allowed
    "1,4-oxazine": "[O;!$(O1-[#6]~[#6]~[#6]-1)]1-C=C-[N;!$(N1-[#6]~[#6]~[#6]-1)]-C=C-1", # not allowed
    "1,2-thiazine": "[S;!$(S1-N-[#6]~[#6]-1)]1-[N;!$(N1-S-[#6]~[#6]-1)]-C=C-C=C-1", # not allowed
    "1,4-thiazine": "[S;!$(S1-[#6]~[#6]~[#6]-1)]1-C=C-[N;!$(N1-[#6]~[#6]~[#6]-1)]-C=C-1", # not allowed
    ## 3 hetero atoms
    "1,2,3-triazine": "[nX2,nX3+]1:[nX2,nX3+]:[nX2,nX3+]:c:c:c:1",
    "1,2,4-triazine": "[nX2,nX3+]1:[nX2,nX3+]:c:[nX2,nX3+]:c:c:1",
    "1,3,5-triazine": "[nX2,nX3+]1:c:[nX2,nX3+]:c:[nX2,nX3+]:c:1",
    # RDKit doesn't consider these to be aromatic (which is correct)
    "1,2,3-oxadiazine":  "O1-[#7;X3]-[#7;!X4]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-1",
    "1,2,4-oxadiazine":  "O1-[#7;!X4]~[#6;X3]~[#7;!X4]-,:[#6;X3]=,:[#6;X3]-1",
    "1,2,5-oxadiazine":  "O1-[#7;X3]-[#6;X3]=,:[#6;X3]-,:[#7;!X4]=,:[#6;X3]-1",
    "1,2,6-oxadiazine":  "O1-[#7;X3]-[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#7;!X4]-1",
    "1,2,3-thiadiazine": "S1-[#7;X3]-[#7;!X4]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-1",
    "1,2,4-thiadiazine": "S1-[#7;!X4]~[#6;X3]~[#7;!X4]-,:[#6;X3]=,:[#6;X3]-1",
    "1,2,5-thiadiazine": "S1-[#7;X3]-[#6;X3]=,:[#6;X3]-,:[#7;!X4]=,:[#6;X3]-1",
    "1,2,6-thiadiazine": "S1-[#7;X3]-[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#7;!X4]-1",
    "1,2,3-oxathiazine": "O1-S-[#7;!X4]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-1",
    "1,2,4-oxathiazine": "O1-S-[#6;X3]=,:[#7;!X4]-,:[#6;X3]=,:[#6;X3]-1",
    "1,2,5-oxathiazine": "O1-S-[#6;X3]=,:[#6;X3]-,:[#7;!X4]=,:[#6;X3]-1",
    "1,2,6-oxathiazine": "O1-S-[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#7;!X4]-1",
    "1,4,2-dioxazine": "O1-[#7;!X4]=,:[#6;X3]-O-[#6;X3]=,:[#6;X3]-1",
    "1,2,3-dithiazine": "S1-S-[#7;!X4]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-1",
    "1,2,4-dithiazine": "S1-S-[#6;X3]=,:[#7;!X4]-,:[#6;X3]=,:[#6;X3]-1",
    "1,4,2-dithiazine": "S1-[#7;!X4]=,:[#6;X3]-O-[#6;X3]=,:[#6;X3]-1",

    ## 4 hetero atoms
    "tetrazine": "c1:n:n:c:n:n:1",

    # 7-membered rings
    # RDKit does not consider these 7-membered heterocycles to be aromatic (which is correct), therefore atomic numbers are used to capture all.
    # -,: (single/aromatic bond) and =,: (double/aromatic bond) is used to allow fusing with aromatic rings
    # the long !$() are to prevent extra inter-ring bonds
    ## 1 hetero atom
    "azepine": "[#7;!r3;!$([#7]1[#6]=[#6]2[#6]=[#6][#6]=[#6]12)]1-[#6;X3;!r3]=[#6;X3;!r3]-[#6;X3;!r3]=[#6;X3;!r3]-[#6;X3;!r3]=[#6;X3;!r3]-1",
    "1-benzazepine": "[#7;!$([#7;R2]12-cc-[#6;R2]~2~[#6]~[#6]~[#6]~1);!$([#7;R2]12-cc-[#6]~[#6;R2]~2~[#6]~[#6]~1);!$([#7;R2]12-cc-[#6]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1-cc-[#6;R2]2~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1-cc-[#6;R2]2~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1-cc-[#6]~[#6;R2]2~[#6]~[#6;R2]~2~1)]1-c2ccccc2-[#6]~[#6]~[#6]~[#6]~1",
    "2-benzazepine": "[#7;!$([#7;R2]12~[#6]-cc-[#6;R2]~2~[#6]~[#6]~1);!$([#7;R2]12~[#6]-cc-[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6;R2]2-cc-[#6;R2]~2~[#6]~[#6]~1);!$([#7]1~[#6;R2]2-cc-[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6;R2]2-cc-[#6]~[#6]~[#6;R2]~2~1);!$([#7]1~[#6]-cc-[#6;R2]2~[#6]~[#6;R2]~2~1)]1~[#6]-c2ccccc2-[#6]~[#6]~[#6]~1",
    "3-benzazepine": "[#7;!$([#7]12~[#6]~[#6]~2-cc-[#6]~[#6]~1);!$([#7]1~[#6]2~[#6]-cc-[#6]~2~[#6]~1);!$([#7]1~[#6]2~[#6]-cc-[#6]~[#6]~2~1);!$([#7]1~[#6]~[#6]2-cc-[#6]~2~[#6]~1)]1~[#6]~[#6]-c2ccccc2-[#6]~[#6]~1",
    "dibenzazepine": "N1-c2ccccc2-[#6;X3]=,:[#6;X3]-c2ccccc2-1",
    "oxepine": "[#8;!r3;!$([#8]1[#6]=[#6]2[#6]=[#6][#6]=[#6]12)]1-[#6;X3;!r3]=,:[#6;X3;!r3]-,:[#6;X3;!r3]=,:[#6;X3;!r3]-,:[#6;X3;!r3]=,:[#6;X3;!r3]-1",
    "thiepine": "[#16!r3;!$([#16]1[#6]=[#6]2[#6]=[#6][#6]=[#6]12)]1-[#6;X3;!r3]=,:[#6;X3;!r3]-,:[#6;X3;!r3]=,:[#6;X3;!r3]-,:[#6;X3;!r3]=,:[#6;X3;!r3]-1",
    ## 2 hetero atoms
    "1,2-diazepine":  "[#7;!$([#7;R2]12~[#7]~[#6;R2]~2~[#6]~[#6]~[#6]~[#6]~1);!$([#7;R2]12~[#7]~[#6]~[#6;R2]~2~[#6]~[#6]~[#6]~1);!$([#7;R2]12~[#7]~[#6]~[#6]~[#6;R2]~2~[#6]~[#6]~1);!$([#7;R2]12~[#7]~[#6]~[#6]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#7]~[#6;R2]2~[#6]~[#6;R2]~2~[#6]~[#6]~1);!$([#7]1~[#7]~[#6;R2]2~[#6]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#7]~[#6;R2]2~[#6]~[#6]~[#6]~[#6;R2]~2~1);!$([#7]1~[#7]~[#6]~[#6;R2]2~[#6]~[#6;R2]~2~[#6]~1)]1~[#7;!$([#7;R2]12~[#7]~[#6;R2]~2~[#6]~[#6]~[#6]~[#6]~1);!$([#7;R2]12~[#7]~[#6]~[#6;R2]~2~[#6]~[#6]~[#6]~1);!$([#7;R2]12~[#7]~[#6]~[#6]~[#6;R2]~2~[#6]~[#6]~1);!$([#7;R2]12~[#7]~[#6]~[#6]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#7]~[#6;R2]2~[#6]~[#6;R2]~2~[#6]~[#6]~1);!$([#7]1~[#7]~[#6;R2]2~[#6]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#7]~[#6;R2]2~[#6]~[#6]~[#6]~[#6;R2]~2~1);!$([#7]1~[#7]~[#6]~[#6;R2]2~[#6]~[#6;R2]~2~[#6]~1)]~[#6]~[#6]~[#6]~[#6]~[#6]~1",
    "1,3-diazepine":  "[#7;!$([#7;R2]12~[#6]~[#7;R2]~2~[#6]~[#6]~[#6]~[#6]~1);!$([#7;R2]12~[#6]~[#7]~[#6;R2]~2~[#6]~[#6]~[#6]~1);!$([#7;R2]12~[#6]~[#7]~[#6]~[#6;R2]~2~[#6]~[#6]~1);!$([#7;R2]12~[#6]~[#7]~[#6]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6;R2]2~[#7]~[#6;R2]~2~[#6]~[#6]~[#6]~1);!$([#7]1~[#6;R2]2~[#7]~[#6]~[#6;R2]~2~[#6]~[#6]~1);!$([#7]1~[#6]~[#7]~[#6;R2]2~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6]~[#7]~[#6;R2]2~[#6]~[#6]~[#6;R2]~2~1)]1~[#6]~[#7;!$([#7;R2]12~[#6]~[#7;R2]~2~[#6]~[#6]~[#6]~[#6]~1);!$([#7;R2]12~[#6]~[#7]~[#6;R2]~2~[#6]~[#6]~[#6]~1);!$([#7;R2]12~[#6]~[#7]~[#6]~[#6;R2]~2~[#6]~[#6]~1);!$([#7;R2]12~[#6]~[#7]~[#6]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6;R2]2~[#7]~[#6;R2]~2~[#6]~[#6]~[#6]~1);!$([#7]1~[#6;R2]2~[#7]~[#6]~[#6;R2]~2~[#6]~[#6]~1);!$([#7]1~[#6]~[#7]~[#6;R2]2~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6]~[#7]~[#6;R2]2~[#6]~[#6]~[#6;R2]~2~1)]~[#6]~[#6]~[#6]~[#6]~1",
    "1,4-diazepine":  "[#7;!$([#7;R2]12~[#6]~[#6;R2]~2~[#7]~[#6]~[#6]~[#6]~1);!$([#7;R2]12~[#6]~[#6]~[#7;R2]~2~[#6]~[#6]~[#6]~1);!$([#7;R2]12~[#6]~[#6]~[#7]~[#6;R2]~2~[#6]~[#6]~1);!$([#7;R2]12~[#6]~[#6]~[#7]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6;R2]2~[#6]~[#7]~[#6;R2]~2~[#6]~[#6]~1);!$([#7]1~[#6;R2]2~[#6]~[#7]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6;R2]2~[#6]~[#7]~[#6]~[#6]~[#6;R2]~2~1);!$([#7]1~[#6]~[#6]~[#7]~[#6;R2]2~[#6]~[#6;R2]~2~1)]1~[#6]~[#6]~[#7;!$([#7;R2]12~[#6]~[#6;R2]~2~[#7]~[#6]~[#6]~[#6]~1);!$([#7;R2]12~[#6]~[#6]~[#7;R2]~2~[#6]~[#6]~[#6]~1);!$([#7;R2]12~[#6]~[#6]~[#7]~[#6;R2]~2~[#6]~[#6]~1);!$([#7;R2]12~[#6]~[#6]~[#7]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6;R2]2~[#6]~[#7]~[#6;R2]~2~[#6]~[#6]~1);!$([#7]1~[#6;R2]2~[#6]~[#7]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6;R2]2~[#6]~[#7]~[#6]~[#6]~[#6;R2]~2~1);!$([#7]1~[#6]~[#6]~[#7]~[#6;R2]2~[#6]~[#6;R2]~2~1)]~[#6;!$(c1ccccc1);!$(c1sccc1)]~[#6;!$(c1ccccc1);!$(c1sccc1)]~[#6]~1", # excludes benzodiazepine, thienodiazepine
    "1,2-oxazepine":  "[#7;!$([#7;R2]12-[OX2]-[#6;R2]~2~[#6]~[#6]~[#6]~[#6]~1);!$([#7;R2]12-[OX2]-[#6]~[#6;R2]~2~[#6]~[#6]~[#6]~1);!$([#7;R2]12-[OX2]-[#6]~[#6]~[#6;R2]~2~[#6]~[#6]~1);!$([#7;R2]12-[OX2]-[#6]~[#6]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1-[OX2]-[#6;R2]2~[#6]~[#6;R2]~2~[#6]~[#6]~1);!$([#7]1-[OX2]-[#6;R2]2~[#6]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1-[OX2]-[#6;R2]2~[#6]~[#6]~[#6]~[#6;R2]~2~1);!$([#7]1-[OX2]-[#6]~[#6;R2]2~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1-[OX2]-[#6]~[#6;R2]2~[#6]~[#6]~[#6;R2]~2~1);!$([#7]1-[OX2]-[#6]~[#6]~[#6;R2]2~[#6]~[#6;R2]~2~1)]1-[OX2]-[#6]~[#6]~[#6]~[#6]~[#6]~1",
    "1,3-oxazepine":  "[#7;!$([#7;R2]12~[#6]-[OX2]-[#6;R2]~2~[#6]~[#6]~[#6]~1);!$([#7;R2]12~[#6]-[OX2]-[#6]~[#6;R2]~2~[#6]~[#6]~1);!$([#7;R2]12~[#6]-[OX2]-[#6]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6;R2]2-[OX2]-[#6;R2]~2~[#6]~[#6]~[#6]~1);!$([#7]1~[#6;R2]2-[OX2]-[#6]~[#6;R2]~2~[#6]~[#6]~1);!$([#7]1~[#6;R2]2-[OX2]-[#6]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6;R2]2-[OX2]-[#6]~[#6]~[#6]~[#6;R2]~2~1);!$([#7]1~[#6]-[OX2]-[#6;R2]2~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6]-[OX2]-[#6;R2]2~[#6]~[#6]~[#6;R2]~2~1);!$([#7]1~[#6]-[OX2]-[#6]~[#6;R2]2~[#6]~[#6;R2]~2~1)]1~[#6]-[OX2]-[#6]~[#6]~[#6]~[#6]~1",
    "1,4-oxazepine":  "[#7;!$([#7;R2]12~[#6]~[#6;R2]~2-[OX2]-[#6]~[#6]~[#6]~1);!$([#7;R2]12~[#6]~[#6]-[OX2]-[#6;R2]~2~[#6]~[#6]~1);!$([#7;R2]12~[#6]~[#6]-[OX2]-[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6;R2]2~[#6]-[OX2]-[#6;R2]~2~[#6]~[#6]~1);!$([#7]1~[#6;R2]2~[#6]-[OX2]-[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6;R2]2~[#6]-[OX2]-[#6]~[#6]~[#6;R2]~2~1);!$([#7]1~[#6]~[#6;R2]2-[OX2]-[#6;R2]~2~[#6]~[#6]~1);!$([#7]1~[#6]~[#6;R2]2-[OX2]-[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6]~[#6;R2]2-[OX2]-[#6]~[#6]~[#6;R2]~2~1);!$([#7]1~[#6]~[#6]-[OX2]-[#6;R2]2~[#6]~[#6;R2]~2~1)]1~[#6]~[#6]-[OX2]-[#6]~[#6]~[#6]~1",
    "1,2-thiazepine":  "[#7;!$([#7;R2]12-S-[#6;R2]~2~[#6]~[#6]~[#6]~[#6]~1);!$([#7;R2]12-S-[#6]~[#6;R2]~2~[#6]~[#6]~[#6]~1);!$([#7;R2]12-S-[#6]~[#6]~[#6;R2]~2~[#6]~[#6]~1);!$([#7;R2]12-S-[#6]~[#6]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1-S-[#6;R2]2~[#6]~[#6;R2]~2~[#6]~[#6]~1);!$([#7]1-S-[#6;R2]2~[#6]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1-S-[#6;R2]2~[#6]~[#6]~[#6]~[#6;R2]~2~1);!$([#7]1-S-[#6]~[#6;R2]2~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1-S-[#6]~[#6;R2]2~[#6]~[#6]~[#6;R2]~2~1);!$([#7]1-S-[#6]~[#6]~[#6;R2]2~[#6]~[#6;R2]~2~1)]1-S-[#6]~[#6]~[#6]~[#6]~[#6]~1",
    "1,3-thiazepine":  "[#7;!$([#7;R2]12~[#6]-S-[#6;R2]~2~[#6]~[#6]~[#6]~1);!$([#7;R2]12~[#6]-S-[#6]~[#6;R2]~2~[#6]~[#6]~1);!$([#7;R2]12~[#6]-S-[#6]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6;R2]2-S-[#6;R2]~2~[#6]~[#6]~[#6]~1);!$([#7]1~[#6;R2]2-S-[#6]~[#6;R2]~2~[#6]~[#6]~1);!$([#7]1~[#6;R2]2-S-[#6]~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6;R2]2-S-[#6]~[#6]~[#6]~[#6;R2]~2~1);!$([#7]1~[#6]-S-[#6;R2]2~[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6]-S-[#6;R2]2~[#6]~[#6]~[#6;R2]~2~1);!$([#7]1~[#6]-S-[#6]~[#6;R2]2~[#6]~[#6;R2]~2~1)]1~[#6]-S-[#6]~[#6]~[#6]~[#6]~1",
    "1,4-thiazepine":  "[#7;!$([#7;R2]12~[#6]~[#6;R2]~2-S-[#6]~[#6]~[#6]~1);!$([#7;R2]12~[#6]~[#6]-S-[#6;R2]~2~[#6]~[#6]~1);!$([#7;R2]12~[#6]~[#6]-S-[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6;R2]2~[#6]-S-[#6;R2]~2~[#6]~[#6]~1);!$([#7]1~[#6;R2]2~[#6]-S-[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6;R2]2~[#6]-S-[#6]~[#6]~[#6;R2]~2~1);!$([#7]1~[#6]~[#6;R2]2-S-[#6;R2]~2~[#6]~[#6]~1);!$([#7]1~[#6]~[#6;R2]2-S-[#6]~[#6;R2]~2~[#6]~1);!$([#7]1~[#6]~[#6;R2]2-S-[#6]~[#6]~[#6;R2]~2~1);!$([#7]1~[#6]~[#6]-S-[#6;R2]2~[#6]~[#6;R2]~2~1)]1~[#6]~[#6]-S-[#6;!$(c1ccccc1)]~[#6;!$(c1ccccc1)]~[#6]~1", # excludes benzothiazepine

    # 5-5 bicyclic
    ## 2 hetero atoms
    ### (1@5 1@5)
    "pyrrolo[2,3-b]pyrrole": "n1:c:c:[cH0]2:c:c:n:[cX3H0]:1:2",
    "pyrrolo[3,2-b]pyrrole": "n1:c:c:[cX3H0]2:n:c:c:[cX3H0]:1:2",
    "pyrrolo[2,3-c]pyrrole": "n1:c:c:[cX3H0]2:c:n:c:[cX3H0]:1:2",
    "pyrrolo[3,4-c]pyrrole": "[#6;X3]1~[#7]~[#6;X3]~[#6;X3]2~[#6;X3]~[#7]~[#6;X3]~[#6;X3]~1~2", # not aromatic
    "furo[2,3-b]furan": "[oX2]1:c:c:[cX3H0]2:c:c:[oX2]:[cX3H0]:1:2",
    "furo[3,2-b]furan": "[oX2]1:c:c:[cX3H0]2:[oX2]:c:c:[cX3H0]:1:2",
    "furo[2,3-c]furan": "[oX2]1:c:c:[cX3H0]2:c:[oX2]:c:[cX3H0]:1:2",
    "thieno[2,3-b]thiophene": "[sX2]1:c:c:[cX3H0]2:c:c:[sX2]:[cX3H0]:1:2",
    "thieno[3,2-b]thiophene": "[sX2]1:c:c:[cX3H0]2:[sX2]:c:c:[cX3H0]:1:2",
    "thieno[2,3-c]thiophene": "[sX2]1:c:c:[cX3H0]2:c:[sX2]:c:[cX3H0]:1:2",
    "furo[2,3-b]pyrrole": "n1:c:c:[cX3H0]2:c:c:[oX2]:[cX3H0]:1:2",
    "furo[3,2-b]pyrrole": "n1:c:c:[cX3H0]2:[oX2]:c:c:[cX3H0]:1:2",
    "furo[3,4-b]pyrrole": "n1:c:c:[cX3H0]2:c:[oX2]:c:[cX3H0]:1:2",
    "furo[2,3-c]pyrrole": "[oX2]1:c:c:[cX3H0]2:c:n:c:[cX3H0]:1:2",
    "thieno[2,3-b]pyrrole": "n1:c:c:[cX3H0]2:c:c:[sX2]:[cX3H0]:1:2",
    "thieno[3,2-b]pyrrole": "n1:c:c:[cX3H0]2:[sX2]:c:c:[cX3H0]:1:2",
    "thieno[3,4-b]pyrrole": "n1:c:c:[cX3H0]2:c:[sX2]:c:[cX3H0]:1:2",
    "thieno[2,3-c]pyrrole": "[sX2]1:c:c:[cX3H0]2:c:n:c:[cX3H0]:1:2",
    "thieno[2,3-b]furan": "[oX2]1:c:c:[cX3H0]2:c:c:[sX2]:[cX3H0]:1:2",
    "thieno[3,2-b]furan": "[oX2]1:c:c:[cX3H0]2:[sX2]:c:c:[cX3H0]:1:2",
    "thieno[3,4-b]furan": "[oX2]1:c:c:[cX3H0]2:c:[sX2]:c:[cX3H0]:1:2",
    "thieno[2,3-c]furan": "c1:[oX2]:c:[cX3H0]2:c:c:[sX2]:[cX3H0]:1:2",
    ### (1@5 1@bridge)
    "pyrrolo[1,2-b]pyrazole": "n1:c:c:[cX3H0]2:c:c:c:n:1:2",
    "pyrrolo[1,2-b]isoxazole": "[oX2]1:c:c:[cX3H0]2:c:c:c:n:1:2",
    "pyrrolo[1,2-b]isothiazole": "[sX2]1:c:c:[cX3H0]2:c:c:c:n:1:2",
    "pyrrolo[2,1-b]imidazole": "n1:c:c:n2:c:c:c:[cX3H0]:1:2",
    "pyrrolo[2,1-b]oxazole": "[oX2]1:c:c:n2:c:c:c:[cX3H0]:1:2",
    "pyrrolo[2,1-b]thiazole": "[sX2]1:c:c:n2:c:c:c:[cX3H0]:1:2",

    # 5-6 bicyclic
    # $([cR2;r5]) means aromatic carbon that is part of 2 rings (R2) and the smallest of them is a 5-membered ring (r5).
    # Excluding it prevents fusing with 5-membered rings (because we have separate SMARTS for 5-6-5 systems) but allows fusing with 6-membered rings (because we don't have a separate 5-6-6 section)
    # !$(c1:a:a:a:a:c:1) prevents fusing with 6-membered rings but allows 5-membered rings
    ## 1 hetero atom
    "indole": "[nX3,nX2-]1:[c;!$(c1:a:a:a:a:c:1)]:[c;!$(c1:a:a:a:a:c:1)]:[cX3H0]2:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5]c(:a)(:a)(:a))]:[cX3H0]:1:2",
    "indolizine": "[c;!$(c1:a:a:a:a:c:1)]1:[c;!$(c1:a:a:a:a:c:1)]:[c;!$(c1:a:a:a:a:c:1)]:[nX3]2:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[cX3H0]:1:2",
    "isoindole": "[c;!$(c1:a:a:a:a:n:1)]1:[nX3,nX2-]:[c;!$(c1:a:a:a:a:n:1)]:[cX3H0]2:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[cX3H0]:1:2",
    "benzofuran": "[oX2]1:[c;!$(c1:a:a:a:a:n:1)]:[c;!$(c1:a:a:a:a:c:1)]:[cX3H0]2:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[cX3H0]:1:2",
    "isobenzofuran": "[c;!$(c(:a)(:a)(:a))]1:[oX2]:[c;!$(c(:a)(:a)(:a))]:[cX3H0]2:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[cX3H0]:1:2",
    "benzothiophene": "[sX2]1:[c;!$(c1:a:a:a:a:c:1)]:[c;!$(c1:a:a:a:a:c:1)]:[cX3H0]2:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[cX3H0]:1:2",
    "isobenzothiophene": "[c;!$(c(:a)(:a)(:a))]1:[sX2]:[c;!$(c(:a)(:a)(:a))]:[cX3H0]2:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[c;!$([cR2;r5](:a)(:a)(:a))]:[cX3H0]:1:2",
    
    ## 2 hetero atoms
    ### (1@5 1@6)
    "4-aza-indole": "[nX3,nX2-]1:[c;!$(c1:a:a:a:a:c:1)]:[c;!$(c1:a:a:a:a:c:1)]:[cX3H0]2:[nX2,nX3+]:c:c:c:[cX3H0]:1:2",
    "5-aza-indole": "[nX3,nX2-]1:[c;!$(c1:a:a:a:a:c:1)]:[c;!$(c1:a:a:a:a:c:1)]:[cX3H0]2:c:[nX2,nX3+]:c:c:[cX3H0]:1:2",
    "6-aza-indole": "[nX3,nX2-]1:[c;!$(c1:a:a:a:a:c:1)]:[c;!$(c1:a:a:a:a:c:1)]:[cX3H0]2:c:c:[nX2,nX3+]:c:[cX3H0]:1:2",
    "7-aza-indole": "[nX3,nX2-]1:[c;!$(c1:a:a:a:a:c:1)]:[c;!$(c1:a:a:a:a:c:1)]:[cX3H0]2:c:c:c:[nX2,nX3+]:[cX3H0]:1:2",
    "4-aza-benzofuran": "[oX2]1:[c;!$(c1:a:a:a:a:c:1)]:[c;!$(c1:a:a:a:a:c:1)]:[cX3H0]2:[nX2,nX3+]:c:c:c:[cX3H0]:1:2",
    "5-aza-benzofuran": "[oX2]1:[c;!$(c1:a:a:a:a:c:1)]:[c;!$(c1:a:a:a:a:c:1)]:[cX3H0]2:c:[nX2,nX3+]:c:c:[cX3H0]:1:2",
    "6-aza-benzofuran": "[oX2]1:[c;!$(c1:a:a:a:a:c:1)]:[c;!$(c1:a:a:a:a:c:1)]:[cX3H0]2:c:c:[nX2,nX3+]:c:[cX3H0]:1:2",
    "7-aza-benzofuran": "[oX2]1:[c;!$(c1:a:a:a:a:c:1)]:[c;!$(c1:a:a:a:a:c:1)]:[cX3H0]2:c:c:c:[nX2,nX3+]:[cX3H0]:1:2",
    "4-aza-benzothiophene": "[sX2]1:[c;!$(c1:a:a:a:a:c:1)]:[c;!$(c1:a:a:a:a:c:1)]:[cX3H0]2:[nX2,nX3+]:c:c:c:[cX3H0]:1:2",
    "5-aza-benzothiophene": "[sX2]1:[c;!$(c1:a:a:a:a:c:1)]:[c;!$(c1:a:a:a:a:c:1)]:[cX3H0]2:c:[nX2,nX3+]:c:c:[cX3H0]:1:2",
    "6-aza-benzothiophene": "[sX2]1:[c;!$(c1:a:a:a:a:c:1)]:[c;!$(c1:a:a:a:a:c:1)]:[cX3H0]2:c:c:[nX2,nX3+]:c:[cX3H0]:1:2",
    "7-aza-benzothiophene": "[sX2]1:[c;!$(c1:a:a:a:a:c:1)]:[c;!$(c1:a:a:a:a:c:1)]:[cX3H0]2:c:c:c:[nX2,nX3+]:[cX3H0]:1:2",
    "4-aza-isoindole": "[c;!$(c1:a:a:a:a:n:1)]1:[nX3,nX2-]:[c;!$(c1:a:a:a:a:n:1)]:[cX3H0]2:[nX2,nX3+]:c:c:c:[cX3H0]:1:2",
    "5-aza-isoindole": "[c;!$(c1:a:a:a:a:n:1)]1:[nX3,nX2-]:[c;!$(c1:a:a:a:a:n:1)]:[cX3H0]2:c:[nX2,nX3+]:c:c:[cX3H0]:1:2",
    "4-aza-isobenzofuran": "[c;!$(c(:a)(:a)(:a))]1:[oX2]:[c;!$(c(:a)(:a)(:a))]:[cX3H0]2:[nX2,nX3+]:c:c:c:[cX3H0]:1:2",
    "5-aza-isobenzofuran": "[c;!$(c(:a)(:a)(:a))]1:[oX2]:[c;!$(c(:a)(:a)(:a))]:[cX3H0]2:c:[nX2,nX3+]:c:c:[cX3H0]:1:2",
    "4-aza-isobenzothiophene": "[c;!$(c(:a)(:a)(:a))]1:[sX2]:[c;!$(c(:a)(:a)(:a))]:[cX3H0]2:[nX2,nX3+]:c:c:c:[cX3H0]:1:2",
    "5-aza-isobenzothiophene": "[c;!$(c(:a)(:a)(:a))]1:[sX2]:[c;!$(c(:a)(:a)(:a))]:[cX3H0]2:c:[nX2,nX3+]:c:c:[cX3H0]:1:2",
    "pyrano[2,3-b]pyrrole": "o1:c:c:c:[cX3H0]2:c:c:[nX2,nX3+]:[cX3H0]:12",
    "pyrano[3,2-b]pyrrole": "o1:c:c:c:[cX3H0]2:[nX2,nX3+]:c:c:[cX3H0]:12",
    "pyrano[3,4-b]pyrrole": "c1:o:c:c:[cX3H0]2:c:c:[nX2,nX3+]:[cX3H0]:12",
    "pyrano[4,3-b]pyrrole": "c1:o:c:c:[cX3H0]2:[nX2,nX3+]:c:c:[cX3H0]:12",
    "pyrano[2,3-c]pyrrole": "o1:c:c:c:[cX3H0]2:c:[nX2,nX3+]:c:[cX3H0]:12",
    "pyrano[3,4-c]pyrrole": "c1:o:c:c:[cX3H0]2:c:[nX2,nX3+]:c:[cX3H0]:12",
    "thiopyrano[2,3-b]pyrrole": "s1:c:c:c:[cX3H0]2:c:c:[nX2,nX3+]:[cX3H0]:12",
    "thiopyrano[3,2-b]pyrrole": "s1:c:c:c:[cX3H0]2:[nX2,nX3+]:c:c:[cX3H0]:12",
    "thiopyrano[3,4-b]pyrrole": "c1:s:c:c:[cX3H0]2:c:c:[nX2,nX3+]:[cX3H0]:12",
    "thiopyrano[4,3-b]pyrrole": "c1:s:c:c:[cX3H0]2:[nX2,nX3+]:c:c:[cX3H0]:12",
    "thiopyrano[2,3-c]pyrrole": "s1:c:c:c:[cX3H0]2:c:[nX2,nX3+]:c:[cX3H0]:12",
    "thiopyrano[3,4-c]pyrrole": "c1:s:c:c:[cX3H0]2:c:[nX2,nX3+]:c:[cX3H0]:12",
    ### (1@5 1@bridge)
    "1-aza-indolizine": "[nX2,nX3+]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[nX3]2:c:c:c:c:[cX3H0]:1:2",
    "2-aza-indolizine": "[c;!$(c1ccccc1)]1:[nX2,nX3+]:[c;!$(c1ccccc1)]:[nX3]2:c:c:c:c:[cX3H0]:1:2",
    "3-aza-indolizine": "[c;!$(c1ccccc1)]1:[c;!$(c1ccccc1)]:[nX2,nX3+]:[nX3]2:c:c:c:c:[cX3H0]:1:2",
    ### (1@6 1@bridge)
    "5-aza-indolizine": "[c;!$(c1ccccc1)]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[nX3]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "6-aza-indolizine": "[c;!$(c1ccccc1)]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[nX3]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "7-aza-indolizine": "[c;!$(c1ccccc1)]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[nX3]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "8-aza-indolizine": "[c;!$(c1ccccc1)]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[nX3]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    ### (2@5 0@6)
    "indazole": "n1:n:[c;!$(c1:a:a:a:a:n:1)]:[cX3H0]2:c:c:c:c:[cX3H0]:1:2", # either tautomer
    "benzimidazole": "[nX3,nX2-]1:[c;!$(c1:a:a:a:a:n:1)]:[nX2,nX3+]:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
    "benzoxazole": "[oX2]1:[c;!$(c1:a:a:a:a:n:1)]:[nX2,nX3+]:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
    "benzisoxazole": "[oX2]1:[nX2,nX3+]:[c;!$(c1:a:a:a:a:n:1)]:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
    "anthranil": "[nX2,nX3+]1:[oX2]:[c;!$(c1:a:a:a:a:n:1)]:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
    "benzothiazole": "[sX2]1:[c;!$(c1:a:a:a:a:n:1)]:[nX2,nX3+]:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
    "benzisothiazole": "[sX2]1:[nX2,nX3+]:[c;!$(c1:a:a:a:a:n:1)]:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
    "benzodioxole": "O1[CX4]O-[cX3H0]2:c:c:c:c:[cX3H0]:2-1",
    "benzodithiole": "[SX2]1[CX4][SX2]-[cX3H0]2:c:c:c:c:[cX3H0]:2-1",

    ## 3 hetero atoms
    ### (3@5 0@6)
    "benzotriazole": "n1:n:n:[cX3H0]2:c:c:c:c:[cX3H0]:1:2", # either tautomer
    "benzo-1,2,3-oxadiazole": "[oX2]1:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
    "benzo-1,2,5-oxadiazole": "[nX2,nX3+]1:[oX2]:[nX2,nX3+]:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
    "benzo-1,2,3-thiadiazole": "[sX2]1:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
    "benzo-1,2,5-thiadiazole": "[nX2,nX3+]1:[sX2]:[nX2,nX3+]:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
    ### (1@5 2@6)
    "4,5-diaza-indole": "[nX3,nX2-]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "4,6-diaza-indole": "[nX3,nX2-]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "4,7-diaza-indole": "[nX3,nX2-]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:[nX2,nX3,nX3+]:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "5,6-diaza-indole": "[nX3,nX2-]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "5,7-diaza-indole": "[nX3,nX2-]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:c:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "6,7-diaza-indole": "[nX3,nX2-]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:c:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "4,5-diaza-benzofuran": "[oX2]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "4,6-diaza-benzofuran": "[oX2]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "4,7-diaza-benzofuran": "[oX2]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:[nX2,nX3,nX3+]:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "5,6-diaza-benzofuran": "[oX2]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "5,7-diaza-benzofuran": "[oX2]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:c:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "6,7-diaza-benzofuran": "[oX2]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:c:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "4,5-diaza-benzothiophene": "[sX2]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "4,6-diaza-benzothiophene": "[sX2]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "4,7-diaza-benzothiophene": "[sX2]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:[nX2,nX3,nX3+]:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "5,6-diaza-benzothiophene": "[sX2]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "5,7-diaza-benzothiophene": "[sX2]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:c:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "6,7-diaza-benzothiophene": "[sX2]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:c:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "4,5-diaza-isoindole": "c1:[nX3,nX2-]:c:[cX3H0]2:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "4,6-diaza-isoindole": "c1:[nX3,nX2-]:c:[cX3H0]2:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "4,7-diaza-isoindole": "c1:[nX3,nX2-]:c:[cX3H0]2:[nX2,nX3,nX3+]:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "5,6-diaza-isoindole": "c1:[nX3,nX2-]:c:[cX3H0]2:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "6,7-diaza-isoindole": "c1:[nX3,nX2-]:c:[cX3H0]2:c:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "4,5-diaza-isobenzofuran": "c1:[oX2]:c:[cX3H0]2:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "4,6-diaza-isobenzofuran": "c1:[oX2]:c:[cX3H0]2:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "4,7-diaza-isobenzofuran": "c1:[oX2]:c:[cX3H0]2:[nX2,nX3,nX3+]:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "5,6-diaza-isobenzofuran": "c1:[oX2]:c:[cX3H0]2:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "6,7-diaza-isobenzofuran": "c1:[oX2]:c:[cX3H0]2:c:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "4,5-diaza-isobenzothiophene": "c1:[sX2]:c:[cX3H0]2:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "4,6-diaza-isobenzothiophene": "c1:[sX2]:c:[cX3H0]2:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "4,7-diaza-isobenzothiophene": "c1:[sX2]:c:[cX3H0]2:[nX2,nX3,nX3+]:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "5,6-diaza-isobenzothiophene": "c1:[sX2]:c:[cX3H0]2:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "6,7-diaza-isobenzothiophene": "c1:[sX2]:c:[cX3H0]2:c:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    ### (2@5 1@6)
    "4-aza-benzimidazole": "[nX3,nX2-]1:c:[nX2,nX3+]:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "5-aza-benzimidazole": "[nX3,nX2-]1:c:[nX2,nX3+]:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "6-aza-benzimidazole": "[nX3,nX2-]1:c:[nX2,nX3+]:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "7-aza-benzimidazole": "[nX3,nX2-]1:c:[nX2,nX3+]:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "4-aza-benzoxazole": "[oX2]1:c:[nX2,nX3+]:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "5-aza-benzoxazole": "[oX2]1:c:[nX2,nX3+]:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "6-aza-benzoxazole": "[oX2]1:c:[nX2,nX3+]:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "7-aza-benzoxazole": "[oX2]1:c:[nX2,nX3+]:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "4-aza-benzothiazole": "[sX2]1:c:[nX2,nX3+]:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "5-aza-benzothiazole": "[sX2]1:c:[nX2,nX3+]:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "6-aza-benzothiazole": "[sX2]1:c:[nX2,nX3+]:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "7-aza-benzothiazole": "[sX2]1:c:[nX2,nX3+]:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "4-aza-indazole": "[nX3,nX2-]1:[nX2,nX3+]:c:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "5-aza-indazole": "[nX3,nX2-]1:[nX2,nX3+]:c:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "6-aza-indazole": "[nX3,nX2-]1:[nX2,nX3+]:c:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "7-aza-indazole": "[nX3,nX2-]1:[nX2,nX3+]:c:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "4-aza-benzisoxazole": "[oX2]1:[nX2,nX3+]:c:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "5-aza-benzisoxazole": "[oX2]1:[nX2,nX3+]:c:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "6-aza-benzisoxazole": "[oX2]1:[nX2,nX3+]:c:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "7-aza-benzisoxazole": "[oX2]1:[nX2,nX3+]:c:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "4-aza-benzisothiazole": "[sX2]1:[nX2,nX3+]:c:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "5-aza-benzisothiazole": "[sX2]1:[nX2,nX3+]:c:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "6-aza-benzisothiazole": "[sX2]1:[nX2,nX3+]:c:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "7-aza-benzisothiazole": "[sX2]1:[nX2,nX3+]:c:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "pyrano[2,3-c]pyrazole": "o1:c:c:c:[cX3H0]2:c:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]:12",
    "pyrano[3,2-c]pyrazole": "o1:c:c:c:[cX3H0]2:[nX2,nX3+]:[nX2,nX3+]:c:[cX3H0]:12",
    "pyrano[3,4-c]pyrazole": "c1:o:c:c:[cX3H0]2:c:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]:12",
    "pyrano[4,3-c]pyrazole": "c1:o:c:c:[cX3H0]2:[nX2,nX3+]:[nX2,nX3+]:c:[cX3H0]:12",
    "thiopyrano[2,3-c]pyrazole": "s1:c:c:c:[cX3H0]2:c:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]:12",
    "thiopyrano[3,2-c]pyrazole": "s1:c:c:c:[cX3H0]2:[nX2,nX3+]:[nX2,nX3+]:c:[cX3H0]:12",
    "thiopyrano[3,4-c]pyrazole": "c1:s:c:c:[cX3H0]2:c:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]:12",
    "thiopyrano[4,3-c]pyrazole": "c1:s:c:c:[cX3H0]2:[nX2,nX3+]:[nX2,nX3+]:c:[cX3H0]:12",
    ### (2@5 1@bridge)
    "1,2-diaza-indolizine": "[nX2,nX3+]1:[nX2,nX3+]:c:[nX3]2:c:c:c:c:[cX3H0]:1:2",
    "1,3-diaza-indolizine": "[nX2,nX3+]1:c:[nX2,nX3+]:[nX3]2:c:c:c:c:[cX3H0]:1:2",
    "2,3-diaza-indolizine": "c1:[nX2,nX3+]:[nX2,nX3+]:[nX3]2:c:c:c:c:[cX3H0]:1:2",
    ### (1@5 1@bridge 1@6)
    "1,5-diaza-indolizine": "[nX2,nX3,nX3+]1:c:c:[nX3]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "2,5-diaza-indolizine": "c1:[nX2,nX3,nX3+]:c:[nX3]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "3,5-diaza-indolizine": "c1:c:[nX2,nX3,nX3+]:[nX3]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "1,6-diaza-indolizine": "[nX2,nX3,nX3+]1:c:c:[nX3]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "2,6-diaza-indolizine": "c1:[nX2,nX3,nX3+]:c:[nX3]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "3,6-diaza-indolizine": "c1:c:[nX2,nX3,nX3+]:[nX3]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "1,7-diaza-indolizine": "[nX2,nX3,nX3+]1:c:c:[nX3]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "2,7-diaza-indolizine": "c1:[nX2,nX3,nX3+]:c:[nX3]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "3,7-diaza-indolizine": "c1:c:[nX2,nX3,nX3+]:[nX3]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "1,8-diaza-indolizine": "[nX2,nX3,nX3+]1:c:c:[nX3]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "2,8-diaza-indolizine": "c1:[nX2,nX3,nX3+]:c:[nX3]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "3,8-diaza-indolizine": "c1:c:[nX2,nX3,nX3+]:[nX3]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    ### (2@6 1@bridge)
    "5,6-diaza-indolizine": "c1:c:c:[nX3]2:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "5,7-diaza-indolizine": "c1:c:c:[nX3]2:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "5,8-diaza-indolizine": "c1:c:c:[nX3]2:[nX2,nX3,nX3+]:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "6,7-diaza-indolizine": "c1:c:c:[nX3]2:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "6,8-diaza-indolizine": "c1:c:c:[nX3]2:c:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "7,8-diaza-indolizine": "c1:c:c:[nX3]2:c:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:[cX3H0]:1:2",

    ## 4 hetero atoms
    ### (2@5 2@6)
    "4,5-diaza-benzimidazole": "[nX3,nX2-]1:c:[nX2,nX3+]:[cX3H0]2:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "purine": "n1:c:n:[cX3H0]2:n:c:n:[c;!$(c~[O,N])]:[cX3H0]:1:2", # either tautomer, excludes adenine, guanine, isoguanine, xanthine, and hypoxanthine
    "4,7-diaza-benzimidazole": "[nX3,nX2-]1:c:[nX2,nX3+]:[cX3H0]2:[nX2,nX3,nX3+]:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "5,6-diaza-benzimidazole": "[nX3,nX2-]1:c:[nX2,nX3+]:[cX3H0]2:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "6,7-diaza-benzimidazole": "[nX3,nX2-]1:c:[nX2,nX3+]:[cX3H0]2:c:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "4,5-diaza-indazole": "[nX3,nX2-]1:[nX2,nX3+]:c:[cX3H0]2:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "4,6-diaza-indazole": "[nX3,nX2-]1:[nX2,nX3+]:c:[cX3H0]2:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "4,7-diaza-indazole": "[nX3,nX2-]1:[nX2,nX3+]:c:[cX3H0]2:[nX2,nX3,nX3+]:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "5,6-diaza-indazole": "[nX3,nX2-]1:[nX2,nX3+]:c:[cX3H0]2:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "5,7-diaza-indazole": "[nX3,nX2-]1:[nX2,nX3+]:c:[cX3H0]2:c:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "6,7-diaza-indazole": "[nX3,nX2-]1:[nX2,nX3+]:c:[cX3H0]2:c:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "4,5-diaza-benzoxazole": "[nX3,nX2-]1:c:[oX2]:[cX3H0]2:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "4,6-diaza-benzoxazole": "[nX3,nX2-]1:c:[oX2]:[cX3H0]2:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "4,7-diaza-benzoxazole": "[nX3,nX2-]1:c:[oX2]:[cX3H0]2:[nX2,nX3,nX3+]:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "5,6-diaza-benzoxazole": "[nX3,nX2-]1:c:[oX2]:[cX3H0]2:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "5,7-diaza-benzoxazole": "[nX3,nX2-]1:c:[oX2]:[cX3H0]2:c:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "6,7-diaza-benzoxazole": "[nX3,nX2-]1:c:[oX2]:[cX3H0]2:c:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "4,5-diaza-benzothiazole": "[nX3,nX2-]1:c:[sX2]:[cX3H0]2:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "4,6-diaza-benzothiazole": "[nX3,nX2-]1:c:[sX2]:[cX3H0]2:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "4,7-diaza-benzothiazole": "[nX3,nX2-]1:c:[sX2]:[cX3H0]2:[nX2,nX3,nX3+]:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "5,6-diaza-benzothiazole": "[nX3,nX2-]1:c:[sX2]:[cX3H0]2:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "5,7-diaza-benzothiazole": "[nX3,nX2-]1:c:[sX2]:[cX3H0]2:c:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "6,7-diaza-benzothiazole": "[nX3,nX2-]1:c:[sX2]:[cX3H0]2:c:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "4,5-diaza-benzisoxazole": "[oX2]1:[nX2,nX3+]:c:[cX3H0]2:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "4,6-diaza-benzisoxazole": "[oX2]1:[nX2,nX3+]:c:[cX3H0]2:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "4,7-diaza-benzisoxazole": "[oX2]1:[nX2,nX3+]:c:[cX3H0]2:[nX2,nX3,nX3+]:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "5,6-diaza-benzisoxazole": "[oX2]1:[nX2,nX3+]:c:[cX3H0]2:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "5,7-diaza-benzisoxazole": "[oX2]1:[nX2,nX3+]:c:[cX3H0]2:c:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "6,7-diaza-benzisoxazole": "[oX2]1:[nX2,nX3+]:c:[cX3H0]2:c:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "4,5-diaza-benzisothiazole": "[sX2]1:[nX2,nX3+]:c:[cX3H0]2:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "4,6-diaza-benzisothiazole": "[sX2]1:[nX2,nX3+]:c:[cX3H0]2:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "4,7-diaza-benzisothiazole": "[sX2]1:[nX2,nX3+]:c:[cX3H0]2:[nX2,nX3,nX3+]:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "5,6-diaza-benzisothiazole": "[sX2]1:[nX2,nX3+]:c:[cX3H0]2:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "5,7-diaza-benzisothiazole": "[sX2]1:[nX2,nX3+]:c:[cX3H0]2:c:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "6,7-diaza-benzisothiazole": "[sX2]1:[nX2,nX3+]:c:[cX3H0]2:c:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    ### (3@5 1@6)
    "4-aza-benzotriazole": "[nX3,nX2-]1:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "5-aza-benzotriazole": "[nX3,nX2-]1:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "6-aza-benzotriazole": "[nX3,nX2-]1:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "7-aza-benzotriazole": "[nX3,nX2-]1:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "4-aza-benzo-1,2,3-oxadiazole": "[oX2]1:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "5-aza-benzo-1,2,3-oxadiazole": "[oX2]1:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "6-aza-benzo-1,2,3-oxadiazole": "[oX2]1:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "7-aza-benzo-1,2,3-oxadiazole": "[oX2]1:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "4-aza-benzo-1,2,5-oxadiazole": "[nX2,nX3+]1:[oX2]:[nX2,nX3+]:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "5-aza-benzo-1,2,5-oxadiazole": "[nX2,nX3+]1:[oX2]:[nX2,nX3+]:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "4-aza-benzo-1,2,3-thiadiazole": "[sX2]1:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "5-aza-benzo-1,2,3-thiadiazole": "[sX2]1:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "6-aza-benzo-1,2,3-thiadiazole": "[sX2]1:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "7-aza-benzo-1,2,3-thiadiazole": "[sX2]1:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "4-aza-benzo-1,2,5-thiadiazole": "[nX2,nX3+]1:[sX2]:[nX2,nX3+]:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "5-aza-benzo-1,2,5-thiadiazole": "[nX2,nX3+]1:[sX2]:[nX2,nX3+]:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    
    # 6-6 bicyclic
    # $([cR2;r6]) means an aromatic carbon that is part of 2 rings (R2) and the smallest of them is a 6-membered ring (r6).
    # Excluding it prevents fusing with 6-membered rings (we have separate SMARTS for 6-6-6 systems) but allows fusing with 5-membered rings (because we don't have a separate 5-6-6 section)
    ## 1 hetero atom
    "quinoline": "[nX2,nX3+]1:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[cX3H0]2:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]:1:2",
    "isoquinoline": "[c;!$(c(:a)(:a)(:a))]1:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[cX3H0]2:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]:1:2",
    "quinolizinium": "c12:c:c:c:c:[n+]:1:c:c:c:c:2",
    ## 2 hetero atoms
    ### (1@6 1@bridge)
    "1-aza-quinolizinium": "c12:n:c:c:c:[n+]:1:c:c:c:c:2",
    "2-aza-quinolizinium": "c12:c:n:c:c:[n+]:1:c:c:c:c:2",
    "3-aza-quinolizinium": "c12:c:c:n:c:[n+]:1:c:c:c:c:2",
    "4-aza-quinolizinium": "c12:c:c:c:n:[n+]:1:c:c:c:c:2",
    ### (2@6 0@6)
    "cinnoline": "[nX2,nX3+]1:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[cX3H0]2:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]:1:2",
    "quinazoline": "[nX2,nX3+]1:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[cX3H0]2:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]:1:2",
    "quinoxaline": "[nX2,nX3+]1:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3+]:[cX3H0]2:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]:1:2",
    "phthalazine": "[c;!$(c(:a)(:a)(:a))]1:[nX2,nX3+]:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[cX3H0]2:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]:1:2",
    "1,4-benzodioxine": "O1-C=C-O-c2:c:c:c:c:c:2-1",
    "1,4-benzodithiine": "[SX2]1-C=C-[SX2]-c2:c:c:c:c:c:2-1",
    "1,4-benzoxathiine": "O1-C=C-[SX2]-c2:c:c:c:c:c:2-1",
    "1,2-benzoxazine": "O1-N-C=C-c2:c:c:c:c:c:2-1",
    "1,4-benzoxazine": "O1-C=C-N-c2:c:c:c:c:c:2-1",
    "1,2-benzothiazine": "[SX2]1-N-C=C-c2:c:c:c:c:c:2-1",
    "1,4-benzothiazine": "[SX2]1-C=C-N-c2:c:c:c:c:c:2-1",
    ### (1@6 1@6)
    "1,5-naphthyridine": "[nX2,nX3+]1:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]2:[nX2,nX3+]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]:1:2",
    "1,6-naphthyridine": "[nX2,nX3+]1:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]2:[c;!$([cR2;r6](:a)(:a)(:a))]:[nX2,nX3+]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]:1:2",
    "1,7-naphthyridine": "[nX2,nX3+]1:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]2:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[nX2,nX3+]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]:1:2",
    "1,8-naphthyridine": "[nX2,nX3+]1:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]2:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[nX2,nX3+]:[cX3H0]:1:2",
    "2,6-naphthyridine": "[c;!$([cR2;r6](:a)(:a)(:a))]1:[nX2,nX3+]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]2:[c;!$([cR2;r6](:a)(:a)(:a))]:[nX2,nX3+]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]:1:2",
    "2,7-naphthyridine": "[c;!$([cR2;r6](:a)(:a)(:a))]1:[nX2,nX3+]:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]2:[c;!$([cR2;r6](:a)(:a)(:a))]:[c;!$([cR2;r6](:a)(:a)(:a))]:[nX2,nX3+]:[c;!$([cR2;r6](:a)(:a)(:a))]:[cX3H0]:1:2",
    ## 3 hetero atoms
    ### (2@6 1@6)
    "5-aza-cinnoline": "[nX2,nX3+]1:[nX2,nX3+]:c:c:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "6-aza-cinnoline": "[nX2,nX3+]1:[nX2,nX3+]:c:c:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "7-aza-cinnoline": "[nX2,nX3+]1:[nX2,nX3+]:c:c:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "8-aza-cinnoline": "[nX2,nX3+]1:[nX2,nX3+]:c:c:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "5-aza-quinazoline": "[nX2,nX3+]1:c:[nX2,nX3+]:c:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "6-aza-quinazoline": "[nX2,nX3+]1:c:[nX2,nX3+]:c:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "7-aza-quinazoline": "[nX2,nX3+]1:c:[nX2,nX3+]:c:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
    "8-aza-quinazoline": "[nX2,nX3+]1:c:[nX2,nX3+]:c:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:2",
    "5-aza-quinoxaline": "[nX2,nX3+]1:c:c:[nX2,nX3+]:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "6-aza-quinoxaline": "[nX2,nX3+]1:c:c:[nX2,nX3+]:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "5-aza-phthalazine": "c1:[nX2,nX3+]:[nX2,nX3+]:c:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:2",
    "6-aza-phthalazine": "c1:[nX2,nX3+]:[nX2,nX3+]:c:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:2",
    "5-aza-1,4-benzodioxine": "O1-[#6;X3]~[#6;X3]-O-c2:c:c:c:n:c:2-1",
    "6-aza-1,4-benzodioxine": "O1-[#6;X3]~[#6;X3]-O-c2:c:c:n:c:c:2-1",
    "5-aza-1,4-benzodithiine": "[SX2]1-[#6;X3]~[#6;X3]-[SX2]-c2:c:c:c:n:c:2-1",
    "6-aza-1,4-benzodithiine": "[SX2]1-[#6;X3]~[#6;X3]-[SX2]-c2:c:c:n:c:c:2-1",
    "5-aza-1,4-benzoxathiine": "O1-[#6;X3]~[#6;X3]-[SX2]-c2:n:c:c:c:c:2-1",
    "6-aza-1,4-benzoxathiine": "O1-[#6;X3]~[#6;X3]-[SX2]-c2:c:n:c:c:c:2-1",
    "7-aza-1,4-benzoxathiine": "O1-[#6;X3]~[#6;X3]-[SX2]-c2:c:c:n:c:c:2-1",
    "8-aza-1,4-benzoxathiine": "O1-[#6;X3]~[#6;X3]-[SX2]-c2:c:c:c:n:c:2-1",
    "5-aza-1,4-benzoxazine": "O1-[#6;X3]~[#6;X3]-N-c2:n:c:c:c:c:2-1",
    "6-aza-1,4-benzoxazine": "O1-[#6;X3]~[#6;X3]-N-c2:c:n:c:c:c:2-1",
    "7-aza-1,4-benzoxazine": "O1-[#6;X3]~[#6;X3]-N-c2:c:c:n:c:c:2-1",
    "8-aza-1,4-benzoxazine": "O1-[#6;X3]~[#6;X3]-N-c2:c:c:c:n:c:2-1",
    "5-aza-1,4-benzothiazine": "[SX2]1-[#6;X3]~[#6;X3]-N-c2:n:c:c:c:c:2-1",
    "6-aza-1,4-benzothiazine": "[SX2]1-[#6;X3]~[#6;X3]-N-c2:c:n:c:c:c:2-1",
    "7-aza-1,4-benzothiazine": "[SX2]1-[#6;X3]~[#6;X3]-N-c2:c:c:n:c:c:2-1",
    "8-aza-1,4-benzothiazine": "[SX2]1-[#6;X3]~[#6;X3]-N-c2:c:c:c:n:c:2-1",
    ## 4 hetero atoms
    ### (2@6 2@6)
    "pteridine": "[nX2,nX3+]1:c:[nX2,nX3+]:c:[cX3H0]2:[nX2,nX3+]:c:c:[nX2,nX3+]:[cX3H0]:1:2",
    "pyrazinopyrazine": "n1:c:c:n:[cX3H0]2:n:c:c:n:[cX3H0]:1:2",
    
    # 5-5-6 tricyclic
    ## 2 hetero atoms
    ### (1@5 1@5 0@6)
    "pyrrolo[2,3-b]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:[cX3H0]3:c:c:c:c:[cX3H0]:3:[nX3,nX2-]:[cX3H0]:1:2",
    "pyrrolo[3,2-b]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:[nX3,nX2-]:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "pyrrolo[3,4-b]indole": "c1:[nX2,nX3+]:c:[cX3H0]2:[cX3H0]3:c:c:c:c:[cX3H0]:3:[nX2,nX3+]:[cX3H0]:1:2",
    "furo[2,3-b]benzofuran": "[oX2]1:c:c:[cX3H0]2:[cX3H0]3:c:c:c:c:[cX3H0]:3:[oX2]:[cX3H0]:1:2",
    "furo[3,2-b]benzofuran": "[oX2]1:c:c:[cX3H0]2:[oX2]:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "thieno[2,3-b]benzothiophene": "[sX2]1:c:c:[cX3H0]2:[cX3H0]3:c:sc:c:c:[cX3H0]:3:[sX2]:[cX3H0]:1:2",
    "thieno[3,2-b]benzothiophene": "[sX2]1:c:c:[cX3H0]2:[sX2]:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "furo[2,3-b]indole": "[oX2]1:c:c:[cX3H0]2:[cX3H0]3:c:c:c:c:[cX3H0]:3:[nX3,nX2-]:[cX3H0]:1:2",
    "furo[3,2-b]indole": "[oX2]1:c:c:[cX3H0]2:[nX3,nX2-]:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "thieno[2,3-b]indole": "[sX2]1:c:c:[cX3H0]2:[cX3H0]3:c:c:c:c:[cX3H0]:3:[nX3,nX2-]:[cX3H0]:1:2",
    "thieno[3,2-b]indole": "[sX2]1:c:c:[cX3H0]2:[nX3,nX2-]:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "pyrrolo[2,3-b]benzofuran": "[nX3,nX2-]1:c:c:[cX3H0]2:[cX3H0]3:c:c:c:c:[cX3H0]:3:[oX2]:[cX3H0]:1:2",
    "pyrrolo[3,2-b]benzofuran": "[nX3,nX2-]1:c:c:[cX3H0]2:[oX2]:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "pyrrolo[2,3-b]benzothiophene": "[nX3,nX2-]1:c:c:[cX3H0]2:[cX3H0]3:c:c:c:c:[cX3H0]:3:[sX2]:[cX3H0]:1:2",
    "pyrrolo[3,2-b]benzothiophene": "[nX3,nX2-]1:c:c:[cX3H0]2:[sX2]:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "thieno[2,3-b]benzofuran": "[sX2]1:c:c:[cX3H0]2:[cX3H0]3:c:c:c:c:[cX3H0]:3:[oX2]:[cX3H0]:1:2",
    "thieno[3,2-b]benzofuran": "[sX2]1:c:c:[cX3H0]2:[oX2]:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "furo[2,3-b]benzothiophene": "[oX2]1:c:c:[cX3H0]2:[cX3H0]3:c:c:c:c:[cX3H0]:3:[sX2]:[cX3H0]:1:2",
    "furo[3,2-b]benzothiophene": "[oX2]1:c:c:[cX3H0]2:[sX2]:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    ### (1@5 1@bridge)
    "pyrrolo[2,3-b]indolizine": "[nX3,nX2-]1:c:c:[cX3H0]2:[nX3]3:c:c:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "pyrrolo[3,2-b]indolizine": "[nX3,nX2-]1:c:c:[cX3H0]2:c:[cX3H0]3:c:c:c:c:[nX3]:3:[cX3H0]:1:2",
    "pyrrolo[3,4-b]indolizine": "c1:[nX3,nX2-]:c:[cX3H0]2:c:[cX3H0]3:c:c:c:c:[nX3]:3:[cX3H0]:1:2",
    "furo[2,3-b]indolizine": "[oX2]1:c:c:[cX3H0]2:[nX3]3:c:c:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "furo[3,2-b]indolizine": "[oX2]1:c:c:[cX3H0]2:c:[cX3H0]3:c:c:c:c:[nX3]:3:[cX3H0]:1:2",
    "furo[3,4-b]indolizine": "c1:[oX2]:c:[cX3H0]2:c:[cX3H0]3:c:c:c:c:[nX3]:3:[cX3H0]:1:2",
    "thieno[2,3-b]indolizine": "[sX2]1:c:c:[cX3H0]2:[nX3]3:c:c:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "thieno[3,2-b]indolizine": "[sX2]1:c:c:[cX3H0]2:c:[cX3H0]3:c:c:c:c:[nX3]:3:[cX3H0]:1:2",
    "thieno[3,4-b]indolizine": "c1:[sX2]:c:[cX3H0]2:c:[cX3H0]3:c:c:c:c:[nX3]:3:[cX3H0]:1:2",
    
    # 5-6-5 tricyclic
    ## 2 hetero atoms
    ### (1@5 0@6 1@5)
    "pyrrolo[2,3-e]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:[cX3H0]3:[nX3,nX2-]:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2", # equivalent to [2,3-g]
    "pyrrolo[3,2-e]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:[cX3H0]3:c:c:[nX3,nX2-]:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "pyrrolo[3,4-e]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:[cX3H0]3:c:[nX3,nX2-]:c:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "pyrrolo[2,3-f]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:c:[cX3H0]3:[nX3,nX2-]:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "pyrrolo[3,2-f]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:c:[cX3H0]3:c:c:[nX3,nX2-]:[cX3H0]:3:c:[cX3H0]:1:2",
    "pyrrolo[3,4-f]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:c:[cX3H0]3:c:[nX3,nX2-]:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "pyrrolo[3,2-g]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:[nX3,nX2-]:[cX3H0]:3:[cX3H0]:1:2",
    "pyrrolo[3,4-g]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:c:c:[cX3H0]3:c:[nX3,nX2-]:c:[cX3H0]:3:[cX3H0]:1:2",
    "pyrrolo[3,4-e]isoindole": "c1:[nX3,nX2-]:c:[cX3H0]2:[cX3H0]3:c:[nX3,nX2-]:c:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "furo[2,3-e]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:[cX3H0]3:[oX2]:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "furo[3,2-e]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:[cX3H0]3:c:c:[oX2]:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "furo[3,4-e]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:[cX3H0]3:c:[oX2]:c:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "furo[2,3-f]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:c:[cX3H0]3:[oX2]:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "furo[3,2-f]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:c:[cX3H0]3:c:c:[oX2]:[cX3H0]:3:c:[cX3H0]:1:2",
    "furo[3,4-f]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:c:[cX3H0]3:c:[oX2]:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "furo[2,3-g]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:c:c:[cX3H0]3:[oX2]:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "furo[3,2-g]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:[oX2]:[cX3H0]:3:[cX3H0]:1:2",
    "furo[3,4-g]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:c:c:[cX3H0]3:c:[oX2]:c:[cX3H0]:3:[cX3H0]:1:2",
    "furo[2,3-e]isoindole": "c1:[nX3,nX2-]:c:[cX3H0]2:[cX3H0]3:[oX2]:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "furo[3,2-e]isoindole": "c1:[nX3,nX2-]:c:[cX3H0]2:[cX3H0]3:c:c:[oX2]:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "furo[3,4-e]isoindole": "c1:[nX3,nX2-]:c:[cX3H0]2:[cX3H0]3:c:[oX2]:c:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "furo[2,3-f]isoindole": "c1:[nX3,nX2-]:c:[cX3H0]2:c:[cX3H0]3:[oX2]:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "thieno[2,3-e]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:[cX3H0]3:[sX2]:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "thieno[3,2-e]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:[cX3H0]3:c:c:[sX2]:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "thieno[3,4-e]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:[cX3H0]3:c:[sX2]:c:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "thieno[2,3-f]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:c:[cX3H0]3:[sX2]:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "thieno[3,2-f]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:c:[cX3H0]3:c:c:[sX2]:[cX3H0]:3:c:[cX3H0]:1:2",
    "thieno[3,4-f]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:c:[cX3H0]3:c:[sX2]:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "thieno[2,3-g]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:c:c:[cX3H0]3:[sX2]:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "thieno[3,2-g]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:[sX2]:[cX3H0]:3:[cX3H0]:1:2",
    "thieno[3,4-g]indole": "[nX3,nX2-]1:c:c:[cX3H0]2:c:c:[cX3H0]3:c:[sX2]:c:[cX3H0]:3:[cX3H0]:1:2",
    "thieno[2,3-e]isoindole": "c1:[nX3,nX2-]:c:[cX3H0]2:[cX3H0]3:[sX2]:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "thieno[3,2-e]isoindole": "c1:[nX3,nX2-]:c:[cX3H0]2:[cX3H0]3:c:c:[sX2]:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "thieno[3,4-e]isoindole": "c1:[nX3,nX2-]:c:[cX3H0]2:[cX3H0]3:c:[sX2]:c:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "thieno[2,3-f]isoindole": "c1:[nX3,nX2-]:c:[cX3H0]2:c:[cX3H0]3:[sX2]:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "furo[2,3-e]benzofuran": "[oX2]1:c:c:[cX3H0]2:[cX3H0]3:[oX2]:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2", # equivalent to [2,3-g]
    "furo[3,2-e]benzofuran": "[oX2]1:c:c:[cX3H0]2:[cX3H0]3:c:c:[oX2]:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "furo[3,4-e]benzofuran": "[oX2]1:c:c:[cX3H0]2:[cX3H0]3:c:[oX2]:c:[cX3H0]:3:c:c:[cX3H0]:1:2",      
    "furo[2,3-f]benzofuran": "[oX2]1:c:c:[cX3H0]2:c:[cX3H0]3:[oX2]:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "furo[3,2-f]benzofuran": "[oX2]1:c:c:[cX3H0]2:c:[cX3H0]3:c:c:[oX2]:[cX3H0]:3:c:[cX3H0]:1:2",
    "furo[3,4-f]benzofuran": "[oX2]1:c:c:[cX3H0]2:c:[cX3H0]3:c:[oX2]:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "furo[3,2-g]benzofuran": "[oX2]1:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:[oX2]:[cX3H0]:3:[cX3H0]:1:2",
    "furo[3,4-g]benzofuran": "[oX2]1:c:c:[cX3H0]2:c:c:[cX3H0]3:c:[oX2]:c:[cX3H0]:3:[cX3H0]:1:2",
    "furo[3,4-e]isobenzofuran": "c1:[oX2]:c:[cX3H0]2:[cX3H0]3:c:[oX2]:c:[cX3H0]:3:c:c:[cX3H0]:1:2",      
    "thieno[2,3-e]benzofuran": "[oX2]1:c:c:[cX3H0]2:[cX3H0]3:[sX2]:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "thieno[3,2-e]benzofuran": "[oX2]1:c:c:[cX3H0]2:[cX3H0]3:c:c:[sX2]:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "thieno[3,4-e]benzofuran": "[oX2]1:c:c:[cX3H0]2:[cX3H0]3:c:[sX2]:c:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "thieno[2,3-f]benzofuran": "[oX2]1:c:c:[cX3H0]2:c:[cX3H0]3:[sX2]:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "thieno[3,2-f]benzofuran": "[oX2]1:c:c:[cX3H0]2:c:[cX3H0]3:c:c:[sX2]:[cX3H0]:3:c:[cX3H0]:1:2",
    "thieno[3,4-f]benzofuran": "[oX2]1:c:c:[cX3H0]2:c:[cX3H0]3:c:[sX2]:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "thieno[2,3-g]benzofuran": "[oX2]1:c:c:[cX3H0]2:c:c:[cX3H0]3:[sX2]:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "thieno[3,2-g]benzofuran": "[oX2]1:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:[sX2]:[cX3H0]:3:[cX3H0]:1:2",
    "thieno[3,4-g]benzofuran": "[oX2]1:c:c:[cX3H0]2:c:c:[cX3H0]3:c:[sX2]:c:[cX3H0]:3:[cX3H0]:1:2",
    "thieno[2,3-e]isobenzofuran": "c1:[oX2]:c:[cX3H0]2:[cX3H0]3:[sX2]:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "thieno[3,2-e]isobenzofuran": "c1:[oX2]:c:[cX3H0]2:[cX3H0]3:c:c:[sX2]:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "thieno[3,4-e]isobenzofuran": "c1:[oX2]:c:[cX3H0]2:[cX3H0]3:c:[sX2]:c:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "thieno[2,3-f]isobenzofuran": "c1:[oX2]:c:[cX3H0]2:c:[cX3H0]3:[sX2]:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "thieno[2,3-e]benzothiophene": "[sX2]1:c:c:[cX3H0]2:[cX3H0]3:[sX2]:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2", # equivalent to [2,3-g]
    "thieno[3,2-e]benzothiophene": "[sX2]1:c:c:[cX3H0]2:[cX3H0]3:c:c:[sX2]:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "thieno[3,4-e]benzothiophene": "[sX2]1:c:c:[cX3H0]2:[cX3H0]3:c:[sX2]:c:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "thieno[2,3-f]benzothiophene": "[sX2]1:c:c:[cX3H0]2:c:[cX3H0]3:[sX2]:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "thieno[3,2-f]benzothiophene": "[sX2]1:c:c:[cX3H0]2:c:[cX3H0]3:c:c:[sX2]:[cX3H0]:3:c:[cX3H0]:1:2",
    "thieno[3,4-f]benzothiophene": "[sX2]1:c:c:[cX3H0]2:c:[cX3H0]3:c:[sX2]:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "thieno[3,2-g]benzothiophene": "[sX2]1:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:[sX2]:[cX3H0]:3:[cX3H0]:1:2",
    "thieno[3,4-g]benzothiophene": "[sX2]1:c:c:[cX3H0]2:c:c:[cX3H0]3:c:[sX2]:c:[cX3H0]:3:[cX3H0]:1:2",
    "thieno[3,4-e]isobenzothiophene": "c1:[sX2]:c:[cX3H0]2:[cX3H0]3:c:[sX2]:c:[cX3H0]:3:c:c:[cX3H0]:1:2",
    ### 1@5 1@bridge
    "pyrrolo[2,3-e]indolizine": "[nX3,nX2-]1:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:c:[nX3]:3:[cX3H0]:1:2",
    "pyrrolo[3,2-e]indolizine": "c1:c:[nX3,nX2-]:[cX3H0]2:c:c:[cX3H0]3:c:c:c:[nX3]:3:[cX3H0]:1:2",
    "pyrrolo[3,4-e]indolizine": "c1:[nX3,nX2-]:c:[cX3H0]2:c:c:[cX3H0]3:c:c:c:[nX3]:3:[cX3H0]:1:2",
    "pyrrolo[2,3-f]indolizine": "[nX3,nX2-]1:c:c:[cX3H0]2:c:[cX3H0]3:c:c:c:[nX3]:3:c:[cX3H0]:1:2",
    "pyrrolo[3,2-f]indolizine": "c1:c:[nX3,nX2-]:[cX3H0]2:c:[cX3H0]3:c:c:c:[nX3]:3:c:[cX3H0]:1:2",
    "pyrrolo[2,3-g]indolizine": "[nX3,nX2-]1:c:c:[cX3H0]2:[cX3H0]3:c:c:c:[nX3]:3:c:c:[cX3H0]:1:2",
    "pyrrolo[3,2-g]indolizine": "c1:c:[nX3,nX2-]:[cX3H0]2:[cX3H0]3:c:c:c:[nX3]:3:c:c:[cX3H0]:1:2",
    "pyrrolo[3,4-g]indolizine": "c1:[nX3,nX2-]:c:[cX3H0]2:[cX3H0]3:c:c:c:[nX3]:3:c:c:[cX3H0]:1:2",
    
    # 6-5-6 tricyclic
    ## 1 hetero atom
    "carbazole": "[nX3,nX2-]1:[cX3H0]2:c:c:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "benzo[a]indolizine": "c1:[nX3]2:c:c:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "benzo[b]indolizine": "c1:[cX3H0]2:c:c:c:c:[nX3]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "dibenzofuran": "[oX2]1:[cX3H0]2:c:c:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "dibenzothiophene": "[sX2]1:[cX3H0]2:c:c:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    ## 2 hetero atoms
    ### (1@6 1@5)
    "1-aza-carbazole": "[nX3,nX2-]1:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3", # alpha-carboline
    "2-aza-carbazole": "[nX3,nX2-]1:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3", # beta-carboline
    "3-aza-carbazole": "[nX3,nX2-]1:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3", # gamma-carboline
    "4-aza-carbazole": "[nX3,nX2-]1:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3", # delta-carboline
    "1-aza-dibenzofuran": "[oX2]1:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "2-aza-dibenzofuran": "[oX2]1:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "3-aza-dibenzofuran": "[oX2]1:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "4-aza-dibenzofuran": "[oX2]1:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "1-aza-dibenzothiophene": "[sX2]1:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "2-aza-dibenzothiophene": "[sX2]1:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "3-aza-dibenzothiophene": "[sX2]1:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "4-aza-dibenzothiophene": "[sX2]1:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    ### (1@6 1@bridge)
    "1-aza-benzo[a]indolizine": "c1:[nX3]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "2-aza-benzo[a]indolizine": "c1:[nX3]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "3-aza-benzo[a]indolizine": "c1:[nX3]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "4-aza-benzo[a]indolizine": "c1:[nX3]2:[nX2,nX3+]:c:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "6-aza-benzo[a]indolizine": "[nX2,nX3+]1:[nX3]2:c:c:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "7-aza-benzo[a]indolizine": "c1:[nX3]2:c:c:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:3",
    "8-aza-benzo[a]indolizine": "c1:[nX3]2:c:c:c:c:[cX3H0]:2:[cX3H0]3:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:3",
    "9-aza-benzo[a]indolizine": "c1:[nX3]2:c:c:c:c:[cX3H0]:2:[cX3H0]3:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:3",
    "10-aza-benzo[a]indolizine": "c1:[nX3]2:c:c:c:c:[cX3H0]:2:[cX3H0]3:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:3",
    "1-aza-benzo[b]indolizine": "c1:[cX3H0]2:c:c:c:c:[nX3]:2:[cX3H0]3:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:3",
    "2-aza-benzo[b]indolizine": "c1:[cX3H0]2:c:c:c:c:[nX3]:2:[cX3H0]3:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:3",
    "3-aza-benzo[b]indolizine": "c1:[cX3H0]2:c:c:c:c:[nX3]:2:[cX3H0]3:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:3",
    "4-aza-benzo[b]indolizine": "c1:[cX3H0]2:c:c:c:c:[nX3]:2:[cX3H0]3:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:3",
    "6-aza-benzo[b]indolizine": "c1:[cX3H0]2:c:c:c:[nX2,nX3+]:[nX3]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "7-aza-benzo[b]indolizine": "c1:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[nX3]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "8-aza-benzo[b]indolizine": "c1:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[nX3]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "9-aza-benzo[b]indolizine": "c1:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[nX3]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "10-aza-benzo[b]indolizine": "[nX2,nX3+]1:[cX3H0]2:c:c:c:c:[nX3]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    ## 3 hetero atoms
    ### 2@6 1@5 0@6
    "1,2-diaza-carbazole": "[nX3,nX2-]1:[cX3H0]2:[nX2,nX3+]:[nX2,nX3+]:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "1,3-diaza-carbazole": "[nX3,nX2-]1:[cX3H0]2:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "1,4-diaza-carbazole": "[nX3,nX2-]1:[cX3H0]2:[nX2,nX3,nX3+]:c:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "2,3-diaza-carbazole": "[nX3,nX2-]1:[cX3H0]2:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "2,4-diaza-carbazole": "[nX3,nX2-]1:[cX3H0]2:c:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "3,4-diaza-carbazole": "[nX3,nX2-]1:[cX3H0]2:c:c:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "1,2-diaza-dibenzofuran": "[oX2]1:[cX3H0]2:c:c:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "1,3-diaza-dibenzofuran": "[oX2]1:[cX3H0]2:c:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "1,4-diaza-dibenzofuran": "[oX2]1:[cX3H0]2:[nX2,nX3,nX3+]:c:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "2,3-diaza-dibenzofuran": "[oX2]1:[cX3H0]2:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "2,4-diaza-dibenzofuran": "[oX2]1:[cX3H0]2:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "3,4-diaza-dibenzofuran": "[oX2]1:[cX3H0]2:[nX2,nX3+]:[nX2,nX3+]:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "1,2-diaza-dibenzothiophene": "[sX2]1:[cX3H0]2:c:c:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "1,3-diaza-dibenzothiophene": "[sX2]1:[cX3H0]2:c:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "1,4-diaza-dibenzothiophene": "[sX2]1:[cX3H0]2:[nX2,nX3,nX3+]:c:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "2,3-diaza-dibenzothiophene": "[sX2]1:[cX3H0]2:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "2,4-diaza-dibenzothiophene": "[sX2]1:[cX3H0]2:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "3,4-diaza-dibenzothiophene": "[sX2]1:[cX3H0]2:[nX2,nX3+]:[nX2,nX3+]:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    ### 1@6 1@5 1@6
    "1,5-diaza-carbazole": "[nX3,nX2-]1:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:2:[cX3H0]3:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:3",
    "1,6-diaza-carbazole": "[nX3,nX2-]1:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:2:[cX3H0]3:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:3",
    "1,7-diaza-carbazole": "[nX3,nX2-]1:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:2:[cX3H0]3:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:3",
    "1,8-diaza-carbazole": "[nX3,nX2-]1:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:3",
    "2,5-diaza-carbazole": "[nX3,nX2-]1:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:2:[cX3H0]3:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:3",
    "2,6-diaza-carbazole": "[nX3,nX2-]1:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:2:[cX3H0]3:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:3",
    "2,7-diaza-carbazole": "[nX3,nX2-]1:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:2:[cX3H0]3:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:3",
    "3,5-diaza-carbazole": "[nX3,nX2-]1:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:2:[cX3H0]3:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:3",
    "3,6-diaza-carbazole": "[nX3,nX2-]1:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:2:[cX3H0]3:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:3",
    "4,5-diaza-carbazole": "[nX3,nX2-]1:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:3",
    "1,6-diaza-dibenzofuran": "[oX2]1:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:3",
    "1,7-diaza-dibenzofuran": "[oX2]1:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:3",
    "1,8-diaza-dibenzofuran": "[oX2]1:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:3",
    "1,9-diaza-dibenzofuran": "[oX2]1:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:3",
    "2,6-diaza-dibenzofuran": "[oX2]1:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:2:[cX3H0]3:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:3",
    "2,7-diaza-dibenzofuran": "[oX2]1:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:2:[cX3H0]3:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:3",
    "2,8-diaza-dibenzofuran": "[oX2]1:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:2:[cX3H0]3:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:3",
    "3,6-diaza-dibenzofuran": "[oX2]1:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:3",
    "3,7-diaza-dibenzofuran": "[oX2]1:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:2:[cX3H0]3:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:3",
    "4,6-diaza-dibenzofuran": "[oX2]1:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:3",
    "1,6-diaza-dibenzothiophene": "[sX2]1:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:3",
    "1,7-diaza-dibenzothiophene": "[sX2]1:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:3",
    "1,8-diaza-dibenzothiophene": "[sX2]1:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:3",
    "1,9-diaza-dibenzothiophene": "[sX2]1:[cX3H0]2:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:2:[cX3H0]3:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:1:3",
    "2,6-diaza-dibenzothiophene": "[sX2]1:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:2:[cX3H0]3:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:3",
    "2,7-diaza-dibenzothiophene": "[sX2]1:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:2:[cX3H0]3:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:3",
    "2,8-diaza-dibenzothiophene": "[sX2]1:[cX3H0]2:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:2:[cX3H0]3:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:3",
    "3,6-diaza-dibenzothiophene": "[sX2]1:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:3",
    "3,7-diaza-dibenzothiophene": "[sX2]1:[cX3H0]2:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:2:[cX3H0]3:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:3",
    "4,6-diaza-dibenzothiophene": "[sX2]1:[cX3H0]2:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:1:3",
    
    # 6-6-6 tricyclic:
    # R2 in [cX3H0;R2] are there to exclude azapyrenes
    ## 1 hetero atom
    "acridine": "[nX2,nX3+]1:[cX3H0]2:c:c:c:c:[cX3H0]:2:c:[cX3H0]3:c:c:c:c:[cX3H0]:1:3", # benzo[b]quinoline
    "xanthene": "O1-[cX3H0]2:c:c:c:c:[cX3H0]:2-[CX4]-[cX3H0]3:c:c:c:c:[cX3H0]:3-1",
    "thioxanthene": "[SX2]1-[cX3H0]2:c:c:c:c:[cX3H0]:2-[CX4]-[cX3H0]3:c:c:c:c:[cX3H0]:3-1",
    "phenanthridine": "[nX2,nX3,nX3+]1:[c;!$(c=O)]:[cX3H0]2:c:c:c:c:[cX3H0;R2]:2:[cX3H0;R2]3:c:c:c:c:[cX3H0]:1:3", # benzo[c]quinoline
    "benzo[f]quinoline": "[nX2,nX3,nX3+]1:c:c:c:[cX3H0;R2]2:[cX3H0;R2]3:c:c:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "benzo[g]quinoline": "[nX2,nX3,nX3+]1:c:c:c:[cX3H0]2:c:[cX3H0]3:c:c:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "benzo[h]quinoline": "[nX2,nX3,nX3+]1:c:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:c:c:[cX3H0;R2]:3:[cX3H0;R2]:1:2",
    "benzo[f]isoquinoline": "c1:[nX2,nX3,nX3+]:c:c:[cX3H0;R2]2:[cX3H0;R2]3:c:c:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2",
    "benzo[g]isoquinoline": "c1:[nX2,nX3,nX3+]:c:c:[cX3H0]2:c:[cX3H0]3:c:c:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "benzo[h]isoquinoline": "c1:[nX2,nX3,nX3+]:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:c:c:[cX3H0;R2]:3:[cX3H0;R2]:1:2",

    ### 2 hetero atoms
    "1,2-diaza-anthracene": "[nX2,nX3+]1:[nX2,nX3,nX3+]:c:c:[cX3H0]2:c:[cX3H0]3:c:c:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "1,3-diaza-anthracene": "[nX2,nX3,nX3+]1:c:[nX2,nX3,nX3+]:c:[cX3H0]2:c:[cX3H0]3:c:c:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "1,4-diaza-anthracene": "[nX2,nX3,nX3+]1:c:c:[nX2,nX3,nX3+]:[cX3H0]2:c:[cX3H0]3:c:c:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "1,5-diaza-anthracene": "[nX2,nX3,nX3+]1:c:c:c:[cX3H0]2:c:[cX3H0]3:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "1,6-diaza-anthracene": "[nX2,nX3,nX3+]1:c:c:c:[cX3H0]2:c:[cX3H0]3:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "1,7-diaza-anthracene": "[nX2,nX3,nX3+]1:c:c:c:[cX3H0]2:c:[cX3H0]3:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "1,8-diaza-anthracene": "[nX2,nX3,nX3+]1:c:c:c:[cX3H0]2:c:[cX3H0]3:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:3:c:[cX3H0]:1:2",
    "1,9-diaza-anthracene": "[nX2,nX3,nX3+]1:c:c:c:[cX3H0]2:c:[cX3H0]3:c:c:c:c:[cX3H0]:3:[nX2,nX3+]:[cX3H0]:1:2",
    "1,10-diaza-anthracene": "[nX2,nX3,nX3+]1:c:c:c:[cX3H0]2:[nX2,nX3+]:[cX3H0]3:c:c:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "2,3-diaza-anthracene": "c1:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:[cX3H0]2:c:[cX3H0]3:c:c:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "2,6-diaza-anthracene": "c1:[nX2,nX3,nX3+]:c:c:[cX3H0]2:c:[cX3H0]3:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "2,7-diaza-anthracene": "c1:[nX2,nX3,nX3+]:c:c:[cX3H0]2:c:[cX3H0]3:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "2,9-diaza-anthracene": "c1:[nX2,nX3,nX3+]:c:c:[cX3H0]2:c:[cX3H0]3:c:c:c:c:[cX3H0]:3:[nX2,nX3+]:[cX3H0]:1:2",
    "2,10-diaza-anthracene": "c1:[nX2,nX3,nX3+]:c:c:[cX3H0]2:[nX2,nX3+]:[cX3H0]3:c:c:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "phenazine": "c1:c:c:c:[cX3H0]2:[nX2,nX3+]:[cX3H0]3:c:c:c:c:[cX3H0]:3:[nX2,nX3+]:[cX3H0]:1:2", # 9,10-diaza-anthracene
    "phenoxazine": "[#8]1~c2:c:c:c:c:c~2~[#7]~c3:c:c:c:c:c~3~1",
    "phenothiazine": "[#16]1~c2:c:c:c:c:c~2~[#7]~c3:c:c:c:c:c~3~1",
    "oxanthrene": "O1-c2:c:c:c:c:c:2-O-c3:c:c:c:c:c:3-1",
    "thianthrene": "S1-c2:c:c:c:c:c:2-S-c3:c:c:c:c:c:3-1",
    "phenoxathiine": "S1-c2:c:c:c:c:c:2-O-c3:c:c:c:c:c:3-1",
    "perimidine": "N1-C=[N,NH+]-[cX3H0]2:c:c:c:[cX3H0]3:c:c:c:[cX3H0]-1:c:2:3",
    "1,2-phenanthroline": "[nX2,nX3+]1:[nX2,nX3,nX3+]:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "1,3-phenanthroline": "[nX2,nX3,nX3+]1:c:[nX2,nX3,nX3+]:c:[cX3H0]2:c:c:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "1,4-phenanthroline": "[nX2,nX3,nX3+]1:c:c:[nX2,nX3,nX3+]:[cX3H0]2:c:c:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "1,5-phenanthroline": "[nX2,nX3,nX3+]1:c:c:c:[cX3H0]2:[nX2,nX3,nX3+]:c:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "1,6-phenanthroline": "[nX2,nX3,nX3+]1:c:c:c:[cX3H0]2:c:[nX2,nX3,nX3+]:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "1,7-phenanthroline": "[nX2,nX3,nX3+]1:c:c:c:[cX3H0]2:c:c:[cX3H0]3:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "1,8-phenanthroline": "[nX2,nX3,nX3+]1:c:c:c:[cX3H0]2:c:c:[cX3H0]3:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "1,9-phenanthroline": "[nX2,nX3,nX3+]1:c:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:3:[cX3H0]:1:2",
    "1,10-phenanthroline": "[nX2,nX3,nX3+]1:c:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:c:[nX2,nX3,nX3+]:[cX3H0]:3:[cX3H0]:1:2",
    "2,3-phenanthroline": "c1:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:c:[cX3H0]2:c:c:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "2,4-phenanthroline": "c1:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:[cX3H0]2:c:c:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "2,5-phenanthroline": "c1:[nX2,nX3,nX3+]:c:c:[cX3H0]2:[nX2,nX3,nX3+]:c:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "2,6-phenanthroline": "c1:[nX2,nX3,nX3+]:c:c:[cX3H0]2:c:[nX2,nX3,nX3+]:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "2,7-phenanthroline": "c1:[nX2,nX3,nX3+]:c:c:[cX3H0]2:c:c:[cX3H0]3:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "2,8-phenanthroline": "c1:[nX2,nX3,nX3+]:c:c:[cX3H0]2:c:c:[cX3H0]3:c:[nX2,nX3,nX3+]:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "2,9-phenanthroline": "c1:[nX2,nX3,nX3+]:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:[nX2,nX3,nX3+]:c:[cX3H0]:3:[cX3H0]:1:2",
    "3,4-phenanthroline": "c1:c:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:[cX3H0]2:c:c:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "3,5-phenanthroline": "c1:c:[nX2,nX3,nX3+]:c:[cX3H0]2:[nX2,nX3,nX3+]:c:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "3,6-phenanthroline": "c1:c:[nX2,nX3,nX3+]:c:[cX3H0]2:c:[nX2,nX3,nX3+]:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "3,7-phenanthroline": "c1:c:[nX2,nX3,nX3+]:c:[cX3H0]2:c:c:[cX3H0]3:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "3,8-phenanthroline": "c1:c:[nX2,nX3,nX3+]:c:[cX3H0]2:c:c:[cX3H0]3:c:[nX2,nX3,nX3+]:c:c:[cX3H0;R2]:3:[cX3H0;R2]:1:2",
    "4,5-phenanthroline": "c1:c:c:[nX2,nX3,nX3+]:[cX3H0]2:[nX2,nX3,nX3+]:c:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "4,6-phenanthroline": "c1:c:c:[nX2,nX3,nX3+]:[cX3H0]2:c:[nX2,nX3,nX3+]:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "4,7-phenanthroline": "c1:c:c:[nX2,nX3,nX3+]:[cX3H0]2:c:c:[cX3H0]3:[nX2,nX3,nX3+]:c:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "5,6-phenanthroline": "c1:c:c:c:[cX3H0]2:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]3:c:c:c:c:[cX3H0]:3:[cX3H0]:1:2",

    # azapyrenes
    "1-aza-pyrene": "n1:c:c:[cX3H0](:[cX3H0]2:[cX3H0]34):c:c:[cX3H0]:4:c:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2",  
    "2-aza-pyrene": "c1:n:c:[cX3H0](:[cX3H0]2:[cX3H0]34):c:c:[cX3H0]:4:c:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2",  
    "4-aza-pyrene": "c1:c:c:[cX3H0](:[cX3H0]2:[cX3H0]34):n:c:[cX3H0]:4:c:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2", 
    "2,7-diaza-pyrene": "c1:n:c:[cX3H0](:[cX3H0]2:[cX3H0]34):c:c:[cX3H0]:4:c:n:c:[cX3H0]:3:c:c:[cX3H0]:1:2",  
    
    # cyclazine
    ## 1 hetero atom
    "cycl[2.2.3]azine": "c1:c:[cX3H0]2:c:c:[cX3H0]3:c:c:c:[cX3H0]:1:[nX3]:2:3",
    "cycl[2.3.3]azine": "c1:c:[cX3H0]2:c:c:[cX3H0]3:c:c:c:[cX3H0]:1:[nX3]:2:3",
    "cycl[3.3.3]azine": "c1:c:c:[cX3H0]2:c:c:c:[cX3H0]3:c:c:c:[cX3H0]:1:[nX3]:2:3",
    ## 2 hetero atoms
    "1-aza-cycl[2.2.3]azine": "[nX2,nX3,nX3+]1:c:[cX3H0]2:c:c:[cX3H0]3:c:c:c:[cX3H0]:1:[nX3]:2:3",
    "2-aza-cycl[2.2.3]azine": "c1:[nX2,nX3,nX3+]:[cX3H0]2:c:c:[cX3H0]3:c:c:c:[cX3H0]:1:[nX3]:2:3",
    "5-aza-cycl[2.2.3]azine": "c1:c:[cX3H0]2:c:c:[cX3H0]3:[nX2,nX3,nX3+]:c:c:[cX3H0]:1:[nX3]:2:3",
    "6-aza-cycl[2.2.3]azine": "c1:c:[cX3H0]2:c:c:[cX3H0]3:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:[nX3]:2:3",

    # miscellaneous
    "porphyrin": "[#6;X3H0]12~[#6]~[#6]~[#6;X3H0](~[#7]~2)~[#6;X3]~[#6;X3H0]3~[#6]~[#6]~[#6;X3H0](~[#7]~3)~[#6;X3]~[#6;X3H0]4~[#6]~[#6]~[#6;X3H0](~[#7]~4)~[#6;X3]~[#6;X3H0]5~[#6]~[#6]~[#6;X3H0](~[#7]~5)~[#6;X3]~1", # including chlorin, bacteriochlorin, and other partial saturations
    "corrin": "[#6]12~[#6]~[#6]~[#6;X3H0](~[#7]~2)~[#6;X3]~[#6;X3H0]3~[#6]~[#6]~[#6;X3H0](~[#7]~3)~[#6;X3]~[#6;X3H0]4~[#6]~[#6]~[#6;X3H0](~[#7]~4)~[#6;X3]~[#6;X3H0]5~[#6]~[#6]~[#6](~[#7]~5)~1", # including corrole, etc. 
    "porphycene": "[#6;X3H0]12~[#6]~[#6]~[#6;X3H0](~[#7]~2)~[#6;X3]~[#6;X3]~[#6;X3H0]3~[#6]~[#6]~[#6;X3H0](~[#7]~3)~[#6;X3H0]4~[#6]~[#6]~[#6;X3H0](~[#7]~4)~[#6;X3]~[#6;X3]~[#6;X3H0]5~[#6]~[#6]~[#6;X3H0](~[#7]~5)~1", # including partial saturations,
    "porphyrazine": "[#6;X3H0]12~[#6]~[#6]~[#6;X3H0](~[#7]~2)~[#7;X2,X3+]~[#6;X3H0]3~[#6]~[#6]~[#6;X3H0](~[#7]~3)~[#7;X2,X3+]~[#6;X3H0]4~[#6]~[#6]~[#6;X3H0](~[#7]~4)~[#7;X2,X3+]~[#6;X3H0]5~[#6]~[#6]~[#6;X3H0](~[#7]~5)~[#7;X2,X3+]~1", # includin phthalocyanin

}

ALIPHATIC_RINGS: Dict[str, str]= {
    # saturated
    "cyclopropane": "C1-C-C-1",
    "cyclobutane":  "[C;!$([C;R2]12CC@2C1)]1-[C;!$([C;R2]12CC@2C1)]-[C;!$([C;R2]12CC@2C1)]-[C;!$([C;R2]12CC@2C1)]-1",
    "cyclopentane": "[C;!$([C;R2]12CC@2CC1)]1-[C;!$([C;R2]12CC@2CC1)]-[C;!$([C;R2]12CC@2CC1)]-[C;!$([C;R2]12CC@2CC1)]-[C;!$([C;R2]12CC@2CC1)]-1",
    "cyclohexane":  "[C;!$([C;R2]12CC@2CCC1);!$([C;R2]12CCC@2CC1)]1-[C;!$([C;R2]12CC@2CCC1);!$([C;R2]12CCC@2CC1)]-[C;!$([C;R2]12CC@2CCC1);!$([C;R2]12CCC@2CC1)]-[C;!$([C;R2]12CC@2CCC1);!$([C;R2]12CCC@2CC1)]-[C;!$([C;R2]12CC@2CCC1);!$([C;R2]12CCC@2CC1)]-[C;!$([C;R2]12CC@2CCC1);!$([C;R2]12CCC@2CC1)]1",
    "cycloheptane": "[C;!$([C;R2]12CC@2CCCC1);!$([C;R2]12CCC@2CCC1)]1-[C;!$([C;R2]12CC@2CCCC1);!$([C;R2]12CCC@2CCC1)]-[C;!$([C;R2]12CC@2CCCC1);!$([C;R2]12CCC@2CCC1)]-[C;!$([C;R2]12CC@2CCCC1);!$([C;R2]12CCC@2CCC1)]-[C;!$([C;R2]12CC@2CCCC1);!$([C;R2]12CCC@2CCC1)]-[C;!$([C;R2]12CC@2CCCC1);!$([C;R2]12CCC@2CCC1)]-[C;!$([C;R2]12CC@2CCCC1);!$([C;R2]12CCC@2CCC1)]-1",
    "cyclooctane":  "[C;!$([C;R2]12CC@2CCCCC1);!$([C;R2]12CCC@2CCCC1);!$([C;R2]12CCCC@2CCC1)]1-[C;!$([C;R2]12CC@2CCCCC1);!$([C;R2]12CCC@2CCCC1);!$([C;R2]12CCCC@2CCC1)]-[C;!$([C;R2]12CC@2CCCCC1);!$([C;R2]12CCC@2CCCC1);!$([C;R2]12CCCC@2CCC1)]-[C;!$([C;R2]12CC@2CCCCC1);!$([C;R2]12CCC@2CCCC1);!$([C;R2]12CCCC@2CCC1)]-[C;!$([C;R2]12CC@2CCCCC1);!$([C;R2]12CCC@2CCCC1);!$([C;R2]12CCCC@2CCC1)]-[C;!$([C;R2]12CC@2CCCCC1);!$([C;R2]12CCC@2CCCC1);!$([C;R2]12CCCC@2CCC1)]-[C;!$([C;R2]12CC@2CCCCC1);!$([C;R2]12CCC@2CCCC1);!$([C;R2]12CCCC@2CCC1)]-[C;!$([C;R2]12CC@2CCCCC1);!$([C;R2]12CCC@2CCCC1);!$([C;R2]12CCCC@2CCC1)]-1",

    # mono unsaturated
    "cyclopropene":  "[CX3]1=[CX3]-C-1",
    "cyclobutene":   "[C;!$([C;R2]12@C@C@2@C@1)]1=[C;!$([C;R2]12@C@C@2@C@1)]-[C;!$([C;R2]12@C@C@2@C@1)]-[C;!$([C;R2]12@C@C@2@C@1)]-1",
    "cyclopentene":  "[C;!$([C;R2]12@C@C@2@C@C@1)]1=[C;!$([C;R2]12@C@C@2@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@1)]-1",
    "cyclohexene":   "[C;!$([C;R2]12@C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]1=[C;!$([C;R2]12@C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]-1",
    "cycloheptene":  "[#6;X3;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]1=,:[#6;X3;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-1",
    "cyclooctene":   "[#6;X3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]1=,:[#6;X3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-1",

    # di unsaturated
    "cyclopentadiene": "[CX3;!$([C;R2]12@C@C@C@2@C@1)]1=[CX3;!$([C;R2]12@C@C@C@2@C@1)]-[CX3;!$([C;R2]12@C@C@C@2@C@1)]=[CX3;!$([C;R2]12@C@C@C@2@C@1)]-[C;!$([C;R2]12@C@C@C@2@C@1)]-1",
    "1,3-cyclohexadiene": "[CX3;!$([C;R2]12C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]1=[CX3;!$([C;R2]12C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]-[CX3;!$([C;R2]12C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]=[CX3;!$([C;R2]12C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]-[CX4;!$([C;R2]12C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]-[C;!$([C;R2]12C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]1", # excludes quinones
    "1,4-cyclohexadiene": "[CX3;!$([C;R2]12C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]1=[CX3;!$([C;R2]12C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]-[CX4;!$([C;R2]12C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]-[CX3;!$([C;R2]12C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]=[CX3;!$([C;R2]12C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]-[C;!$([C;R2]12C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]1", # excludes quinones
    "1,3-cycloheptadiene":  "[CX3;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]1=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-1",
    "1,4-cycloheptadiene":  "[CX3;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]1=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-1",
    "1,3-cyclooctadiene":   "[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]1=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-1",
    "1,4-cyclooctadiene":   "[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]1=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-1",
    "1,5-cyclooctadiene":   "[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]1=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-1",

    # tri unsaturated
    "cycloheptatriene":  "[CX3;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]1=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[CX3;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[CX3;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]1",
    "1,3,5-cyclooctatriene":  "[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]1=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]1",
    "1,3,6-cyclooctatriene":  "[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]1=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]1",

    # tetra unsaturated
    "cyclooctatetraene": "[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]1=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]=[CX3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]1",

    # bicyclic
    "spiro_carbon": "[CD4;R2;x4]",
    "indane": "c1:c:c:c:[cX3H0]2-[C;!$([C;R2]12CC@2cc1)]-[C;!$([C;R2]12CC@2cc1)]-[C;!$([C;R2]12CC@2cc1)]-[cX3H0]:1:2",
    "tetralin": "c1:c:c:c:[cX3H0]2-CCCC-[cX3H0]:1:2",
    "decalin": "C12CCCCC1CCCC2",
    "norbornane": "C12CCC(C1)CC2",
    "norbornene": "C12[#6;X3]=,:[#6;X3]C(C1)CC2",
    "norbornadiene": "C12[#6;X3]=,:[#6;X3]C(C1)[#6;X3]=,:[#6;X3]2",
    "norpinane": "C12CCCC(C1)C2",
    "2-norpinene": "C12[#6;X3]=,:[#6;X3]CC(C1)C2",

    # 3+ rings
    "adamantane": "C12CC(C3)CC(C2)CC3C1",
    "steroid_rings": "[#6]1~[#6]~[#6]~[#6,#7]~[#6]2~[#6]~[#6]~[#6]3~[#6]4~[#6]~[#6]~[#6]~[#6]~4~[#6]~[#6]~[#6]~3~[#6]~2~1", # including 4-aza steroids
}

HETEROALIPHATIC_RINGS: Dict[str, str] = {
    # !$(C=[O,S]) next to heteroatoms prevents OXO / THIOXO variants
    # !$(C=[N,O,S]) is used next to nitrogens to also exclude amidines and guanidines

    # 3-membered
    "epoxide": "O1[CX4][CX4]1",
    "aziridine": "N1[CX4][CX4]1",
    "azirine": "N1=[CX3][CX4]1",
    "episulfide": "[SX2]1[CX4][CX4]1",
    "episelenide": "[Se]1[CX4][CX4]1",
    "oxaziridine": "O1N[CX4]1",

    # 4-membered
    "azetidine": "N1-[C;!$(C=O)]-C-[C;!$(C=O)]-1", # excludes beta-lactams
    "azetine": "N1[#6;X3]=,:[#6;X3]-C-1",
    "oxetane": "O1-C-C-C-1",
    "oxetine": "O1[#6;X3]=,:[#6;X3]-C-1",
    "1,3-dioxetane": "O1-C-O-C-1",
    "thietane": "S1-C-C-C-1",
    "1,2-dithietane": "S1-S-C-C-1",
    "1,3-dithietane": "S1-C-S-C-1",

    # 5-membered
    "pyrrolidine": "[N;!$(N12[#6]~[#6]~[#6]C2~[#6]~[#6]~[#6]~1);!$(N12[#6]~[#6]~[#6]C2~[#6]~[#6]~[#6]~[#6]~1)]1-[C;!$(C=[N,O,S])]-C-C-[C;!$(C=[N,O,S])]-1", # excludes pyrrolizidine, indolizidine
    "1-pyrroline": "N1=[CX3]-C-C-[C;!$(C=O)]-1",
    "2-pyrroline": "N1-[CX3]=[CX3]-C-[C;!$(C=O)]-1", # excludes indoline
    "3-pyrroline": "N1-[C;!$(C=O)]-[CX3]=[CX3]-[C;!$(C=O)]-1",
    "pyrazolidine": "N1-N-[C;!$(C=O)]-C-[C;!$(C=O)]-1",
    "1-pyrazoline": "[NX2,NX3+]1=[NX2,NX3+]-[C;!$(C=O)]-C-[C;!$(C=O)]-1",
    "2-pyrazoline": "N1-[NX2,NX3+]=[CX3]-C-[C;!$(C=O)]-1",
    "imidazolidine": "N1-[C;!$(C=O)]-N-C-[C;!$(C=O)]-1",
    "3-imidazoline": "N1-[C;!$(C=O)]-[NX2,NX3+]=[CX3]-[C;!$(C=O)]-1",
    "4-imidazoline": "N1-[CX4;!$(C=O)]-N-[CX3]=[CX3]-1",
    "oxolane": "O1-[C;!$(C=[O,S])]-C-C-[C;!$(C=[O,S])]-1",
    "benzo[b]oxolane": "O1-c2ccccc2-[CX4]-[CX4]-1",
    "benzo[c]oxolane": "O1-[CX4]-c2ccccc2-[CX4]-1",
    "2-oxolene": "O1-[CX3]=[CX3]-C-[C;!$(C=[O,S])]-1",
    "3-oxolene": "O1-[C;!$(C=[O,S])]-[CX3]=[CX3]-[C;!$(C=[O,S])]-1",
    "thiolane": "[SX2]1-[C;!$(C=[O,S])]-C-C-[C;!$(C=[O,S])]-1",
    "benzo[b]thiolane": "[SX2]1-c2ccccc2-[CX4]-[CX4]-1",
    "benzo[c]thiolane": "[SX2]1-[CX4]-c2ccccc2-[CX4]-1",
    "2-thiolene": "[SX2]1-[CX3]=[CX3]-C-[C;!$(C=[O,S])]-1",
    "3-thiolene": "[SX2]1-[C;!$(C=[O,S])]-[CX3]=[CX3]-[C;!$(C=[O,S])]-1",
    "sulfolane": "[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]1-C-C-C-C-1",
    "2-sulfolene": "[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]1-[#6;X3]=,:[#6;X3]-C-C-1",
    "3-sulfolene": "[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]1-C-[#6;X3]=,:[#6;X3]-C-1",
    "1,3-dioxolane": "O1-[C;!$(C=[O,S])]-O-C-[C;!$(C=[O,S])]-1",
    "1,3-dioxole": "O1-[C;!$(C=[O,S])]-O-[CX3]=[CX3]-1",
    "1,2-dithiolane": "[SX2]1-[SX2]-C-C-[C;!$(C=[O,S])]-1",
    "1,3-dithiolane": "[SX2]1-[C;!$(C=[O,S])]-[SX2]-C-[C;!$(C=[O,S])]-1",
    "1,2-dithiole": "[SX2]1-[SX2]-[C;!$(C=[O,S])]-[CX3]=[CX3]-1",
    "1,3-dithiole": "[SX2]1-[C;!$(C=[O,S])]-[SX2]-[CX3]=[CX3]-1",
    "1,3-oxathiolane": "[SX2]1-[C;!$(C=[O,S])]-O-C-[C;!$(C=[O,S])]-1",
    "1,3-oxathiole": "[SX2]1-[C;!$(C=[O,S])]-O-[CX3]=[CX3]-1",
    "oxazolidine": "N1-[C;!$(C=[O,S])]-O-C-[C;!$(C=[O,S])]-1",
    "2-oxazoline": "[NX2,NX3+]1=[CX3]-O-[C;!$(C=[O,S])]-[C;!$(C=[O,S])]-1",
    "3-oxazoline": "N1-[C;!$(C=[O,S])]-O-[C;!$(C=[O,S])]-[CX3]=1",
    "4-oxazoline": "N1-[C;!$(C=[O,S])]-O-[CX3]=[CX3]-1",

    # 6-membered
    "piperidine": "[#7;!$(N12C[#6]~[#6]([#6]~[#6]2)[#6]~[#6]1);!$(N12[#6]~[#6]~[#6]~[#6]C2~[#6]~[#6]~[#6]~1);!$(N12[#6]~[#6]~[#6]~[#6]C2~[#6]~[#6]~[#6]~[#6]~1);!$([#7]1-C2-[#6]~[#6]-C(-[#6]~1)-[#6]~[#6]-2)]1-,:[#6]-C-C-C-[#6]-,:1", # excludes quinuclidine, isoquinuclidine, indolizidine, quinolizidine
    "piperazine": "[#7]1-,:[#6]-,:[#6]-[N;!$(N12-[#6]~[#6]-N(-[#6]~[#6]-2)-[#6]~[#6]-1)]-C-C-1",
    "1,4-dihydropyridine": "[#7]1-[#6]=,:[#6]-[CX4]-[#6]=,:[#6]-1",
    "oxane": "O1-[C;!$(C=[O,S])]-C-C-C-[C;!$(C=[O,S])]-1",
    "1,2-dioxane": "O1-O-[C;!$(C=[O,S])]-C-C-[C;!$(C=[O,S])]-1",
    "1,3-dioxane": "O1-[C;!$(C=[O,S])]-O-[C;!$(C=[O,S])]-C-[C;!$(C=[O,S])]-1",
    "1,4-dioxane": "O1-[C;!$(C=[O,S])]-[C;!$(C=[O,S])]-O-[C;!$(C=[O,S])]-[C;!$(C=[O,S])]-1",
    "thiane": "[SX2]1-[C;!$(C=[O,S])]-C-C-C-[C;!$(C=[O,S])]-1",
    "1,2-dithiane": "[SX2]1[SX2]-[C;!$(C=[O,S])]-C-C-[C;!$(C=[O,S])]-1",
    "1,3-dithiane": "[SX2]1-[C;!$(C=[O,S])]-[SX2]-[C;!$(C=[O,S])]-C-[C;!$(C=[O,S])]-1",
    "1,4-dithiane": "[SX2]1-[C;!$(C=[O,S])]-[C;!$(C=[O,S])]-[SX2]-[C;!$(C=[O,S])]-[C;!$(C=[O,S])]-1",
    "morpholine": "O1-[C;!$(C=[O,S])]-[C;!$(C=[N,O,S])]-[#7]-,:[#6;!$(C=[N,O,S])]-,:[#6;!$(C=[O,S])]-1",
    "thiomorpholine": "[SX2]1-[C;!$(C=[O,S])]-[C;!$(C=[N,O,S])]-[#7]-,:[#6;!$(C=[N,O,S])]-,:[#6;!$(C=[O,S])]-1",
}

OXO_RINGS: Dict[str, str] = {

    # 3-membered
    "cyclopropenone": "O=[c;!$(c1c([O,N])c([O,N])1)]1cc1",
    
    # 4-membered
    "cyclobutenedione": "O=[c;!$(c1cc([O,N,S])c([O,N,S])1)]1c(=O)cc1",
    "β-lactam": "O=C1-N-[#6;!$(C=O)]~[#6]-1",

    # 5-membered
    "indenone": "O=C1-[#6;X3;!$(c1ccccc1)]~[#6;X3;!$(c1ccccc1)]-[cX3H0]2:c:c:c:c:[cX3H0]:2-1",
    "fluorenone": "O=C1-[cX3H0]2:c:c:c:c:[cX3H0]:2-[cX3H0]3:c:c:c:c:[cX3H0]:3-1",
    "2-pyrrolin-4-one": "O=C1[CX4]N[#6;X3]~[#6;X3]1",
    "2-pyrrolin-5-one": "O=C1-N[#6;X3;!$(c1ccccc1)]~[#6;X3;!$(c1ccccc1)][CX4]1", # excludes oxindole
    "oxindole": "O=C1-N-c2ccccc2-[C;!$([CD4;R2;x4])]1", # excludes spiro-oxindole
    "3-pyrrolinone": "O=C1-N[CX4][#6;X3]~[#6;X3]1",
    "3-pyrazolone": "O=c1:n:n:c:c:1",
    "4-imidazolinone": "O=c1n[c;!$(c1ccccc1)][c;!$(c1ccccc1)]n1",
    "benzimidazolinone": "O=c1nc(cccc2)c2n1",
    "hydantoin": "O=C1-N-C(=O)-[#7]~[#6;!$(C=O)]1",
    "oxazolidinone": "O=C1N[CX4][CX4]O1",
    "oxazolidinedione": "O=C1NC(=O)[CX4]O1",
    "2-thiazolidinone": "O=C1N[CX4][CX4][SX2]1",
    "4-thiazolidinone": "O=C1N[CX4][SX2][CX4]1",
    "thiazolidindione": "O=C1NC(=O)[CX4][SX2]1",

    # 6-membered
    "1,2-benzoquinone": "O=C1-C(=O)-[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-1",
    "1,4-benzoquinone": "O=C1-[#6;X3]=,:[#6;X3]-C(=O)-[#6;X3]=,:[#6;X3]-1",
    "1,2-quinone_methide": "O=C1-C(=C)-[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-1",
    "1,2-quinone_dimethide": "C=c1c(=C)cccc1",
    "1,4-benzoquinone_methide": "O=C1-[#6;X3]=,:[#6;X3]-C(=O)-[#6;X3]=,:[#6;X3]-1",
    "1,4-quinone_methide": "O=C1-[#6;X3]=,:[#6;X3]-C(=C)-[#6;X3]=,:[#6;X3]-1",
    "1,4-quinone_dimethide": "C=c1ccc(=C)cc1",
    "1,5-naphthoquinone": "O=C1-[#6;X3]=,:[#6;X3]-,:[#6;X3]=[#6;X3]2-C(=O)-[#6;X3]=,:[#6;X3]-,:[#6;X3]=[#6;X3]12",
    "1,7-naphthoquinone": "O=C1-[#6;X3]=,:[#6;X3]-,:[#6;X3]=[#6;X3]2-[#6;X3]=,:[#6;X3]-C(=O)-[#6;X3]=[#6;X3]12",
    "2,6-naphthoquinone": "O=C1-[#6;X3]=,:[#6;X3]-[#6;X3]2=[#6;X3]-C(=O)-[#6;X3]=,:[#6;X3]-[#6;X3]-2=[#6;X3]-1",
    "2-pyrone": "O=c1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:o1", # excludes benzopyrone
    "4-pyrone": "O=c1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:o:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]1", # excludes benzopyrone
    "1-benzo[c]pyrone": "O=c1:o:c:c:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
    "2-benzo[b]pyrone": "o1:c(=O):c:c:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
    "3-benzo[c]pyrone": "c1:o:c(=O):c:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
    "4-benzo[b]pyrone": "o1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:c(=O):[cX3H0]2:c:c:c:c:[cX3H0]:1:2", # excludes xanthone
    "xanthone": "o1:[cX3H0]2:c:c:c:c:[cX3H0]:2:c(=O):[cX3H0]3:c:c:c:c:[cX3H0]:3:1",
    "thioxanthone": "o1:[cX3H0]2:c:c:c:c:[cX3H0]:2:c(=[SX1]):[cX3H0]3:c:c:c:c:[cX3H0]:3:1",
    "2-pyridone": "O=c1:n:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:1", # excludes quinolones, etc.
    "4-pyridone": "O=c1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:n:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:1", # excludes quinolones, etc.
    "1-isoquinolone": "O=c1:n:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
    "2-quinolone":    "n1:c(=O):[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
    "3-isoquinolone": "c1:n:c(=O):c:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
    "4-quinolone":    "n1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:c(=O):[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
    "acridone":       "n1:[cX3H0]2:c:c:c:c:[cX3H0]:2:c(=O):[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "phenanthridone": "n1:c(=O):[cX3H0]2:c:c:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
    "2-pyrimidone": "O=c1:n:[c;!$(c~[O,N])]:[c;!$(c1ncnc1)]:[c;!$(c~[O,N]);!$(c1ncnc1)]:n:1", # excludes nucleobases
    "4-pyrimidone": "O=c1:n:[c;!$(c~[O,N])]:n:[c;!$(c1ncnc1);!$(c1ccccc1)]:[c;!$(c1ncnc1);!$(c1ccccc1)]:1", # excludes nucleobases, quinazolin-4-one
    "pyridazinone": "O=c1:n:n:[c;!$(c=O)]:c:c:1", # excludes pyridazinedione
    "pyridazine-3,6-dione": "O=c1:n:n:c(=O):c:c:1",
    "pyrazinone": "O=c1:n:c:c:n:[c;!$(c=O)]:1", # excludes pyrazinedione
    "pyrazine-2,3-dione": "O=c1:n:c:c:n:c(=O):1", 
    "1,2-oxazin-4-one": "O=c1:c:n:o:c:c:1",
    "1,2-oxazin-6-one": "O=c1:o:n:c:c:c:1",
    "1,3-oxazin-2-one": "O=c1:o:c:c:c:n:1",
    "1,3-oxazin-4-one": "O=c1:c:c:o:c:n:1",
    "1,3-oxazin-6-one": "O=c1:o:c:n:c:c:1",
    "1,4-oxazin-2-one": "O=c1:o:c:c:n:c:1",
    "1,2-thiazin-4-one": "O=c1:c:n:s:c:c:1",
    "1,2-thiazin-6-one": "O=c1:s:n:c:c:c:1",
    "1,3-thiazin-2-one": "O=c1:s:c:c:c:n:1",
    "1,3-thiazin-4-one": "O=c1:c:c:s:c:n:1",
    "1,3-thiazin-6-one": "O=c1:s:c:n:c:c:1",
    "1,4-thiazin-2-one": "O=c1:s:c:c:n:c:1",
    "barbiturate": "[OX1,OH]~[#6;X3H0]1~N~[#6;X3H0](~[OX1,OH])~N~[#6;X3H0](~[OX1,OH])-[CX4]-1",

    # pteridines
    "pteridin-2-one": "n1:c(=O):n:[c;!$(c=O);!$(c-N)]:[cX3H0]2:n:[c;!$(c=O)]:[c;!$(c=O)]:n:[cX3H0]:12", # excludes isopterin
    "pteridin-4-one": "n1:[c;!$(c=O);!$(c-N)]:n:c(=O):[cX3H0]2:n:[c;!$(c=O)]:[c;!$(c=O)]:n:[cX3H0]:12", # excludes pterin
    "pteridin-6-one": "n1:[c;!$(c=O)]:n:[c;!$(c=O)]:[cX3H0]2:n:c(=O):[c;!$(c=O)]:n:[cX3H0]:12",
    "pteridin-7-one": "n1:[c;!$(c=O)]:n:[c;!$(c=O)]:[cX3H0]2:n:[c;!$(c=O)]:c(=O):n:[cX3H0]:12",
    "pterin": "n1:c(-N):n:c(=O):[cX3H0]2:n:[c;!$(c=O)]:[c;!$(c=O)]:n:[cX3H0]:12",
    "tetrahydropterin": "n1:c(-N):n:c(=O):[cX3H0]2-N-C-C-N-[cX3H0]:12",
    "isopterin": "n1:c(=O):n:c(-N):[cX3H0]2:n:[c;!$(c=O)]:[c;!$(c=O)]:n:[cX3H0]:12",
    "xanthopterin": "n1:c(-N):n:c(=O):[cX3H0]2:n:c(=O):[c;!$(c=O)]:n:[cX3H0]:12", 
    "isoxanthopterin": "n1:c(-N):n:c(=O):[cX3H0]2:n:[c;!$(c=O)]:c(=O):n:[cX3H0]:12",
    "leucopterin": "n1:c(-N):n:c(=O):[cX3H0]2:n:c(=O):c(=O):n:[cX3H0]:12",
    "pteridine-2,4-dione": "n1:c(=O):n:c(=O):[cX3H0]2:n:[c;!$(c1ccccc1);!$(c=O)]:[c;!$(c1ccccc1);!$(c=O)]:n:[cX3H0]:12", # excludes alloxazine, isoalloxazine
    "pteridine-2,6-dione": "n1:c(=O):n:[c;!$(c=O)]:[cX3H0]2:n:c(=O):[c;!$(c=O)]:n:[cX3H0]:12",
    "pteridine-2,7-dione": "n1:c(=O):n:[c;!$(c=O)]:[cX3H0]2:n:[c;!$(c=O)]:c(=O):n:[cX3H0]:12",
    "pteridine-4,6-dione": "n1:[c;!$(c=O)]:n:c(=O):[cX3H0]2:n:c(=O):[c;!$(c=O)]:n:[cX3H0]:12",
    "pteridine-4,7-dione": "n1:[c;!$(c=O)]:n:c(=O):[cX3H0]2:n:c(=O):[c;!$(c=O)]:n:[cX3H0]:12",
    "alloxazine":    "[nX3]1:c(=O):[nX3]:c(=O):[cX3H0]2:[nX2,nX3+]:c3:c:c:c:c:c:3:[nX2,nX3+]:[cX3H0]:12",
    "isoalloxazine": "[nX2,nX3+]1:c(=O):[nX3]:c(=O):[cX3H0]2:[nX2,nX3+]:c3:c:c:c:c:c:3:[nX3]:[cX3H0]:12",
    "5,10-dihydro-alloxazine": "[nX3]1:c(=O):[nX3]:c(=O):[cX3H0]2-N-c3:c:c:c:c:c:3-N-[cX3H0]:1:2",
   
   # 7-membered
    "tropone": "[OX1]~c1:[c;!$(c~O)]:c:c:c:c:[c;!$(c~O)]:1", # excludes tropolone
    "tropolone": "O~c1:c(O):c:c:c:c:c:1"

}

BIOMOLECULES: Dict[str, str] = {
    # amino acids
    # N-terminus can be attached to anything, C-terminus cannot be aldehydes or ketones
    ## proteinogenic 
    "glycine": "N-[CH2]-C(=[O;!$(O=C1NC(=O)NC1)])-[!$([#6,#1])]", # excludes hydantoin
    "alanine": "N-[CH](-[CH3])-C(=O)-[!$([#6,#1])]",
    "serine": "N-[CH](-[CH2]O)-C(=O)-[!$([#6,#1])]",
    "cysteine": "N-[CH](-[CH2]S)-C(=O)-[!$([#6,#1])]",
    "selenocysteine": "N-[CH](-[CH2][Se])-C(=O)-[!$([#6,#1])]",
    "methionine": "N-[CH](-[CH2][CH2]S[CH3])-C(=O)-[!$([#6,#1])]",
    "valine": "N-[CH](-[CH]([CH3])[CH3])-C(=O)-[!$([#6,#1])]",
    "threonine": "N-[CH](-[CH](O)[CH3])-C(=O)-[!$([#6,#1])]",
    "leucine": "N-[CH](-[CH2][CH]([CH3])[CH3])-C(=O)-[!$([#6,#1])]",
    "isoleucine": "N-[CH](-[CH]([CH3])[CH2][CH3])-C(=O)-[!$([#6,#1])]",
    "phenylalanine": "N-[CH](-[CH2]-c1:c:c:[c;!$(c-O)]:c:c:1)-C(=O)-[!$([#6,#1])]",
    "tyrosine": "N-[CH](-[CH2][c]1:c:c:c(-O):c:c:1)-C(=O)-[!$([#6,#1])]",
    "histidine": "N-[CH](-[CH2][c]1:n:[cH]:n:[cH]:1)-C(=O)-[!$([#6,#1])]",
    "tryptophan": "N-[CH](-[CH2]c1:c2:c:c:c:c:c:2:n:c:1)-C(=O)-[!$([#6,#1])]",
    "aspartic_acid": "N-[CH](-[CH2][C](=O)O)-C(=O)-[!$([#6,#1])]",
    "asparagine": "N-[CH](-[CH2][C](=O)N)-C(=O)-[!$([#6,#1])]",
    "glutamic_acid": "N-[CH](-[CH2][CH2][C](=O)O)-C(=O)-[!$([#6,#1])]",
    "glutamine": "N-[CH](-[CH2][CH2][C](=O)N)-C(=O)-[!$([#6,#1])]",
    "lysine": "N-[CH](-[CH2][CH2][CH2][CH2][#7])-C(=O)-[!$([#6,#1])]",
    "arginine": "N-[CH](-[CH2][CH2][CH2]N~[CX3H0](~N)~N)-C(=O)-[!$([#6,#1])]",
    "proline":  "N1-[CH](-[CH2][CH2][CH2]1)-C(=O)-[!$([#6,#1])]",
    "4-hydroxyproline": "N1-[CH](-[CH2][CH](-O)[CH2]1)-C(=O)-[!$([#6,#1])]",
    ## non-proteinogenic
    "homoalanine": "N-[CH](-[CH2][CH3])-C(=O)-[!$([#6,#1])]",
    "norvaline": "N-[CH](-[CH2][CH2][CH3])-C(=O)-[!$([#6,#1])]",
    "homoserine": "N-[CH](-[CH2][CH2]O)-C(=O)-[!$([#6,#1])]",
    "homocysteine": "N-[CH](-[CH2][CH2]S)-C(=O)-[!$([#6,#1])]",
    "penicillamine": "N-[CH](-C([CH3])([CH3])S)-C(=O)-[!$([#6,#1])]",
    "selenomethionine": "N-[CH](-[CH2][CH2][Se][CH3])-C(=O)-[!$([#6,#1])]",
    "ornithine": "N-[CH](-[CH2][CH2][CH2][#7;!$(N-C(=[N,O])-N)])-C(=O)-[!$([#6,#1])]",
    "citrulline": "N-[CH](-[CH2][CH2][CH2]N-C(=O)-N)-C(=O)-[!$([#6,#1])]",
    
    # nucleobases
    ## purine bases
    "adenine": "N~[cX3H0]1:n:[c;!$(c~O)]:n:[cX3H0]2:n:c:n:[cX3H0]:1:2", # many tautomers, excludes isoguanine
    "guanine": "O~[cX3H0]1:n:[cX3H0](~N):n:[cX3H0]2:n:c:n:[cX3H0]:1:2", # many tautomers 
    "isoguanine": "N~[cX3H0]1:n:[cX3H0](~O):n:[cX3H0]2:n:c:n:[cX3H0]:1:2", # many tautomers
    "hypoxanthine": "O~[cX3H0]1:n:[c;!$(c~[O,N])]:n:[cX3H0]2:n:c:n:[cX3H0]:1:2", # many tautomers, excludes guanine and xanthine
    "xanthine": "O~[cX3H0]1:n:c(~O):n:[cX3H0]2:n:[c;!$(c~O)]:n:[cX3H0]:1:2", # many tautomers
    "uric acid": "O~[cX3H0]1:n:c(~O):n:[cX3H0]2:n:c(~O):n:[cX3H0]:1:2", # many tautomers
    ## pyrimidine bases
    "cytosine":    "[OX1,OH]~[cX3H0]1:n:[cX3H0](~N):[c;!$(c1ncnc1);!$(c1nccnc1)]:[c;!$(c1ncnc1);!$(c1nccnc1);!$(c~O)]:n:1", # many tautomers, excludes purines, isopterin
    "isocytosine": "[OX1,OH]~[cX3H0]1:n:[cX3H0](~N):n:[c;!$(c1ncnc1);!$(c1nccnc1);!$(c1NCCNc1)]:[c;!$(c1ncnc1);!$(c1nccnc1);!$(c1NCCNc1)]:1", # many tautomers, excludes purines, pterin
    "uracil": "[OX1,OH]~[cX3H0]1:n:[cX3H0;!$(c1nccnc1)](~[OX1,OH]):[c;!$(c-C);!$(c1ncnc1);!$(c1nccnc1);!$(c1NccNc1)]:[c;!$(c~O);!$(c1ncnc1)]:n:1", # many tautomers, excludes purines, pteridines
    "thymine": "[OX1,OH]~[cX3H0]1:n:[cX3H0](~[OX1,OH]):c(-C):c:n:1", # many tautomers 

    # monosaccharides & related
    ## triose
    "glyceraldehyde": "O-[CH2]-[CH](-O)-[CH]=O",
    "glycerol": "O-[CH2]-[CH](-O)-[CH2]-O",
    ## tetrose
    "aldotetrose": "[$([CH]=O),$([CH](-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH2](O)", #including hemiacetal, acetal, etc. and cyclic forms
    "ketotetrose": "O[CH2][$(C=O),$(C(-[O,S,N,n])-[O,S,N,n])][CH](O)[CH2](O)", #including hemiketal, ketal, etc, and cyclic forms
    "tetritol": "O-[CH2]-[CH](-O)-[CH](-O)-[CH2]-O",
    ## pentose
    "aldopentose": "[$([CH]=O),$([CH](-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH](O)[CH2](O)", #including hemiacetal, acetal, etc. and cyclic forms
    "ketopentose": "O[CH2][$(C=O),$(C(-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH2](O)", #including hemiketal, ketal, etc, and cyclic forms
    "pentitol": "O-[CH2]-[CH](-O)-[CH](-O)-[CH](-O)-[CH2]-O",
    ## hexose
    "aldohexose": "[$([CH]=O),$([CH](-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH](O)[CH](O)[CH2](O)", #including hemiacetal, acetal, etc. and cyclic forms
    "ketohexose": "O[CH2][$(C=O),$(C(-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH](O)[CH2](O)", #including hemiketal, ketal, etc. and cyclic forms
    "hexitol": "O-[CH2]-[CH](-O)-[CH](-O)-[CH](-O)-[CH](-O)-[CH2]-O",
    "inositol": "O-[CH]1-[CH](-O)-[CH](-O)-[CH](-O)-[CH](-O)-[CH]1(-O)",
    ## heptose
    "aldoheptose": "[$([CH]=O),$([CH](-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH](O)[CH](O)[CH](O)[CH2](O)", #including hemiacetal, acetal, etc. and cyclic forms
    "ketoheptose": "O[CH2][$(C=O),$(C(-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH](O)[CH](O)[CH2](O)", #including hemiketal, ketal, etc. and cyclic forms
    "heptitol": "O-[CH2]-[CH](-O)-[CH](-O)-[CH](-O)-[CH](-O)-[CH](-O)-[CH2]-O",
    
    # fats
    ## lipids
    "1-monoglyceride": "[O-,OH,OH2+]-[CH2]-[CH](-[O-,OH,OH2+])-[CH2]-O-C(=O)-[#6]",
    "2-monoglyceride": "[O-,OH,OH2+]-[CH2]-[CH](-O-C(=O)-[#6])-[CH2]-[O-,OH,OH2+]",
    "1,2-diglyceride": "[O-,OH,OH2+]-[CH2]-[CH](-O-C(=O)-[#6])-[CH2]-O-C(=O)-[#6]",
    "1,3-diglyceride": "[#6]-C(=O)-O-[CH2]-[CH](-[O-,OH,OH2+])-[CH2]-O-C(=O)-[#6]",
    "triglyceride": "[#6]-C(=O)-O-[CH2]-[CH](-O-C(=O)-[#6])-[CH2]-O-C(=O)-[#6]",
    "phosphatidyl": "O-[$([PX4]=O),$([PX4+]-[O-])](-O)-O-[CH2]-[CH](-O-C(=O)-[#6])-[CH2]-O-C(=O)-[#6]",
    "plasmanyl_phospholipid": "O-[$([PX4]=O),$([PX4+]-[O-])](-O)-O-[CH2]-[CH](-O-C(=O)-[#6])-[CH2]-O-[CH2]-[CH2]",
    "plasmenyl_phospholipid": "O-[$([PX4]=O),$([PX4+]-[O-])](-O)-O-[CH2]-[CH](-O-C(=O)-[#6])-[CH2]-O-[CH]=[CH]",
    "sphinganine": "O-[CH2]-[CH](-N)-[CH](-O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]",
    "sphingosine": "O-[CH2]-[CH](-N)-[CH](-O)-[CH]=[CH]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]",
    "phytosphingosine": "O-[CH2]-[CH](-N)-[CH](-O)-[CH](-O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]",
    ## fatty acids
    ### saturated
    "lauroyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 12:0
    "tridecoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 13:0
    "myristoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 14:0
    "pentadecoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 15:0
    "palmitoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 16:0
    "margaroyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 17:0
    "stearoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 18:0
    "nonadecoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 19:0
    "arachidoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 20:0 
    ### unsaturated (all cis)
    "palmitoleoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-\[CH]=[CH]/-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 16:1 omega-7
    "sapienoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-\[CH]=[CH]/-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 16:1 omega-10
    "oleoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-\[CH]=[CH]/-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 18:1 omega-9
    "linoleoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 18:2 omega-6
    "alpha-linoleoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-[CH3]", # 18:3 omega-3
    "gamma-linoleoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 18:3 omega-6
    "stearidonoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-[CH]=[CH]-[CH2]-[CH2]=[CH2]-[CH2]-[CH3]", # 18:4  omega-3
    "dihomo-gamma-linoleoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 20:3  omega-6
    "arachidonoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 20:4  omega-6
    "eicosapentaenoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-[CH3]", # 20:5  omega-3
    "adrenoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 22:4  omega-6
    "docosapentaenoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-[CH3]", # 22:5  omega-3
    "docosahexaenoyl": "[*]-C(=O)-[CH2]-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-\[CH]=[CH]/-[CH2]-[CH3]", # 22:6  omega-3
    ### trans fatty acids
    "trans-palmitoleoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-/[CH]=[CH]/-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # trans 16:1 omega-7
    "elaidoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-/[CH]=[CH]/-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # trans 18:1 omega-9
    "vaccenoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-/[CH]=[CH]/-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # trans 18:1 omega-7
    "rumenoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-\[CH]=[CH]/-/[CH]=[CH]/-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # trans11 18:2 omega-7
    "linoleladoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-/[CH]=[CH]/-[CH2]-/[CH]=[CH]/-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # trans,trans 18:2 omega-6
    
    # alkaloid/privileged scaffolds
    "phenethylamine": "[N;!$(N1ccCC1);!$(N1CccCC1);!$(N1~C~CC2ccCC(C2)1);!$(N1~C~C~C~C2~C1~C-c3cnc4cccc-2c34)]-[C;!$([CH](N)(C(=O)-[!#6;!#1])-[CH2]-c1ccccc1)]-C-c1ccccc1", # excludes phenylalanine, tyrosine, indoline, tetrahydroisoquinoline, 6,7-benzomorphan, ergoline
    "tryptamine": "[N;!$(N1~C~C~C~C2~C1~C-c3cnc4cccc-2c34);!$(N1~C~C-c2c3ccccc3nc2-C~C~1)]-[C;!$([CH](N)(C(=O)-[!#6;!#1])-[CH2]-c1c2ccccc2nc1)]-C-c1c2ccccc2nc1", # excludes tryptophan, ergoline, ibogalog
    "indoline": "N1-[C;!$(C=O)]-C-c2ccccc2-1", # excludes oxindole, cyclotryptamine
    "cyclotryptamine": "N1-C2-N-C-C-C-C-2-c3ccccc3-1",
    "spiro-oxindole": "N1-C(=O)-[CD4;R2;x4]-c2ccccc2-1", 
    "tetrahydroisoquinoline": "C1-N-C-C-c2ccccc2-1",
    "pyrrolizidine": "N12-,=[#6]-,=,:[#6]-,=,:[#6]-,=C-,=2-,=[#6]-,=,:[#6]-,=,:[#6]-,=1",
    "indolizidine": "N12-,=[#6]-,=,:[#6]-,=,:[#6]-,=C-,=2-,=[#6]-,=,:[#6]-,=,:[#6]-,=,:[#6]-,=1",
    "quinolizidine": "N12-,=[#6]-,=,:[#6]-,=,:[#6]-,=,:[#6]-,=C-,=2-,=[#6]-,=,:[#6]-,=,:[#6]-,=,:[#6]-,=1",
    "quinuclidine": "N12-[#6]-,=,:[#6]-C(-[#6]-,=,:[#6]-2)-[#6]-,=,:[#6]-1",
    "isoquinuclidine": "[#6]12-[#7]-,=,:[#6]-C(-[#6]-,=,:[#6]-2)-[#6]-,=,:[#6]-1",
    "tropane": "[CH3]-N1-C2-[#6]-,=,:[#6]-,=,:[#6]-C(-[#6]-,=,:[#6]-2)-1",
    "DABCO": "N12-[#6]-,=,:[#6]-N(-[#6]-,=,:[#6]-2)-[#6]-,=,:[#6]-1",    
    "granatane": "[CH3]-N1-C2-[#6]-,=,:[#6]-,=,:[#6]-C(-[#6]-,=,:[#6]-,=,:[#6]-2)-1", 
    "morphan": "C12-[#7]-,=,:[#6;!$(c1ccccc1)]-,=,:[#6;!$(c1ccccc1)]-C(-C2)-[#6;!$(c1ccccc1)]-,=,:[#6;!$(c1ccccc1)]-,=,:[#6]-1",
    "3,4-benzomorphan": "C12-[#7]-,:c3ccccc3-C(-C2)-[#6]-,=,:[#6]-,=,:[#6]-1",
    "6,7-benzomorphan": "C12-[#7]-,=,:[#6]-,=,:[#6]-[C;!$(C123-[#6]~[#6]~[#6]~[#6]-C-2-C(-[#7]~[#6]~[#6]-3)-C-cc-1)](-C2)-c3ccccc3-,:[#6]-1", # excludes morphinan
    "morphinan": "C12-[#7]-,=,:[#6]-,=,:[#6]-C(-C3-2)(-[#6]-,=,:[#6]-,=,:[#6]-,=,:[#6]-3)-c4ccccc4-,:[#6]-1",
    "aporphine": "N1-C-C-c2cccc(c23)~c4ccccc4~[#6]~[#6]~3-1",
    "ergoline": "n1cc(-C-,=C3-,=N-,=,:[#6]-,=,:[#6]-,=,:[#6]-,=C4-,=3)c2c-4cccc12",
    "ibogalog": "[#7]1-,=,:[#6]-,=,:[#6]-c2c3ccccc3nc2-[#6]-,=,:[#6]-,=,:1",
    "quinazolin-4-one":  "O=c1:n:c:n:c2ccccc:1:2",
    "protoberberine": "c1ccccc-c2cc3ccccc3c[n+]2-C-C-1",
    "benzodiazepine": "[#7]1-c2ccccc2-[#6]~[#7]~[#6]~[#6]~1",
    "thienodiazepine": "[#7]1-c2sccc2-[#6]~[#7]~[#6]~[#6]~1",
    "benzothiazepine": "[#7]1-c2ccccc2-S-[#6]~[#6]~[#6]~1",

    # misc
    "choline": "O-[CH2]-[CH2]-[N+](-[CH3])(-[CH3])-[CH3]",
    "taurine": "O-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[CH2]-[CH2]-N",

}


ALL = CORE | BRANCHES | MAIN_GROUP | HOMOAROMATICS | HETEROAROMATICS | ALIPHATIC_RINGS | HETEROALIPHATIC_RINGS | OXO_RINGS | BIOMOLECULES
