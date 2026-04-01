# -*- coding: utf-8 -*-

"""Presets for SMARTS and molecular substructure matching."""

from typing import Dict

__all__ = [
    "ALL",
    "CORE",
    "BRANCHES",
    "HOMOAROMATICS",
    "HETEROAROMATICS",
    "ALIPHATIC_RINGS",
    "HETEROALIPHATIC_RINGS",
    "OXO_RINGS",
    "THIOXO_RINGS",
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
    "internal_alkyne": "[CH0]#[CH0]",
    "allene": "[#6X3]=C=[#6X3]",
    "enyne": "[!$(C#C)]-C(-[!$(C#C)])=C(-[!$(C#C)])-C#C",
    "enediyne": "C#C-C(-[!$(C#C)])=C(-[!$(C#C)])-C#C",
    "carbene": "[*]-[CX2]-[*]",
    # Oxygen
    ## 1x O
    "primary_alcohol": "[CX4;H2]-[O-,OH,OH2+]",
    "secondary_alcohol": "[CX4;H;!$(C(O)(O))]-[O-,OH,OH2+]",
    "tertiary_alcohol": "[CX4;H0;!$(C(O)(O))]-[O-,OH,OH2+]",
    "allylic_hydroxy": "[O-,OH,OH2+]-[$([CX4;!$(C(O)(O))]-C=C)]",
    "benzylic_hydroxy": "[O-,OH,OH2+]-[$([CX4;!$(C(O)(O))]-a1aaaaa1)]",
    "aromatic_hydroxy": "[cX3H0]-[O-,OH,OH2+]", 
    "geminal_diol": "[CX4](-[O-,OH,OH2+])-[O-,OH,OH2+]",
    "vicinal_diol": "[OH]-[CX4]-[CX4]-[OH]",
    "enol": "[#6]=C-[OH,O-,OH2+]",
    "ether": "[#6;!$(C#C);!$(C=[C,O,S,N]);!$(C#N)]-[OX2;!r3;!$(O1-[#6;X3]~[#6;X3]~[O,S,N]~[#6;X3]~[#6;X3]-1)]-[#6;!$(C#C);!$(C=[C,O,S,N]);!$(C#N)]", # excludes enol ethers, ynol ethers, epoxides, esters, anhydrides, oxines, cyanates
    "enol_ether": "[CX3;$(C=C);!$(C(O)O)]-O-[#6;!$(C=O);!$(C#N)]",
    "enol_ester": "[CX3;$(C=C)]-O-C=O",
    "ynol_ether": "[CX2;$(C#C)]-O-[#6;!$(C=O);!$(C#N)]",
    "ynol_ester": "[CX2;$(C#C)]-O-C=O",
    "aldehyde": "[#6]-[CH](=O)",
    "ketone": "[#6;!$(C#N)]-C(=O)-[#6;!$(C#N)]", # excludes acyl cyanides
    "ketene": "C=C=O",
    "conjugated_carbonyl": "O=[C;$(C-C=C)]", # 'O=C-C=C' is not used; it would double count the carbonyl if it's conjugated on both sides
    "oxonium": "[!#1]-[OX3+](-[!#1])-[!#1]",

    ## 2x O
    "methylenedioxy": "[*]-O-[CH2;!$([CX4]1Oc2ccccc2O1)]-O-[*]", # excludes benzodioxoles
    "peroxide": "[!#1;!$(C=O)][#8][#8][!#1;!$(C=O)]", # excludes mono- and diacyl peroxides
    "hydroperoxy": "[!#1;!$(C=O)]-O-[O-,OH,OH2+]", # excludes peracids
    "aliphatic_carboxylic_acid": "C-C(=O)-[OH,O-,OH2+]",
    "aromatic_carboxylic_acid": "c-C(=O)-[OH,O-,OH2+]",
    "carboxylate_ester": "[#1,#6]-C(=O)-[O;!$(O1[#6](=O)[#6]1);!$(O1[#6](=O)[#6]~[#6]1);!$(O1[#6](=O)[#6]~[#6]~[#6]1);!$(O1[#6](=O)[#6]~[#6]~[#6]~[#6]1)]-[#6;!$(C=[O,S])]", # excludes lactones upto delta
    "hemiacetal": "[O-,OH,OH2+]-[CX4;!$(C(O)(O)[O,S,N,n])]-O-[!#1;!$(C=O)]", # including hemiketals
    "acetal": "[!#1;!$(C=O)]-O-[CX4;!$(C(O)(O)([O,S,N,n]));!H2;!$([CX4]1Oc2ccccc2O1)]-O-[!#1;!$(C=O)]", # including ketals, excluding acylals, methylenedioxy
    "ketene_acetal": "C=C(-O)-O",
    "acetylenediolate": "[!#1]-O-C#C-O-[!#1]",

    ## 3x O
    "percarboxylic_acid": "[#6]-C(=O)-O-[O-,OH,OH2+]",
    "percarboxylate_ester": "[#6]-C(=O)-O-O-[#6;!$(C=O)]",
    "carbonate": "[*]-O-C(=O)-O-[*]",
    "carboxylic_anhydride": "[*]-C(=O)-O-C(=O)-[*]",
    "orthoester": "[#6,#1]-C(-O)(-O)-O",
    "monothio_orthoester": "[#6,#1]-C(-O)(-O)-[SX2]",
    "dithio_orthoester": "[#6,#1]-C(-O)(-[SX2])-[SX2]",
    "trithio_orthoester": "[#6,#1]-C(-[SX2])(-[SX2])-[SX2]",
    "ozonide": "O1-[CX4]-O-O-[CX4]-1",
    "alpha-keto_acid": "[#6,#1]-C(=O)-C(=O)-[OH,O-,OH2+]",
    "alpha-keto_ester": "[#6,#1]-C(=O)-C(=O)-O-[#6,#14;!$(C=[O,S])]",
    "hemiacylal": "C(=O)-O-[CX4;!$(C(O)(O)[O,S,N,n])]-[O-,OH,OH2+]",
    "O-acyl_hemiacetal": "C(=O)-O-[CX4;!$(C(O)(O)[O,S,N,n])]-O-[!#1;!$(C=O)]",
    "deltate": "O=c1c(O)c1(O)",
    "squarate": "O=c1c(=O)c(O)c1O",
    ## 4+ O
    "diacyl_peroxide": "[#6](=O)[#8][#8][#6](=O)",
    "orthocarbonate": "O-[CX4](-O)(-O)-O",
    "monothio_orthocarbonate": "O-[CX4](-O)(-O)-[SX2]",
    "dithio_orthocarbonate": "O-[CX4](-O)(-[SX2])-[SX2]",
    "trithio_orthocarbonate": "O-[CX4](-[SX2])(-[SX2])-[SX2]",
    "tetrathio_orthocarbonate": "[SX2]-[CX4](-[SX2])(-[SX2])-[SX2]",
    "acylal": "C(=O)-O-[CX4;!$(C(O)(O)[O,S,N,n])]-O-C(=O)",

    # Nitrogen
    ## 1x N
    "primary_amine": "[NX3H2,NX4H3+]-[#6;!$(C=[O,S,N])]", # excludes anillines, amides, etc.
    "aryl_amine": "[$(N-[c;!r3;!r4](:[!$(c=O)]):[!$(c=O)]);!$(N=[O,S,N,P])]",
    "secondary_amine": "[#6;!$(C=[O,S,N])]-[NX3H,NX4H2+;!r3;!$(N1~[#6;X3]~[#6;X3]~[O,S]~[#6;X3]~[#6;X3]-1)]-[#6;!$(C=[O,S,N])]", # excludes amides, aziridines, oxazines, thiazines
    "tertiary_amine": "[#6;!$(C=[O,S,N])]-[NX3H0,NX4H+;!r3;!$(N1~[#6;X3]~[#6;X3]~[O,S]~[#6;X3]~[#6;X3]-1)](-[#6;!$(C=[O,S,N])])-[#6;!$(C=[O,S,N])]", # excludes amides, aziridines, oxazines, thiazines
    "quaternary_ammonium": "[!$([#6,#8,#5;-])]-[NX4H0+](-[!$([#6,#8,#5;-])])(-[!$([#6,#8,#5;-])])-[!$([#6,#8,#5;-])]", # excludes N-oxides; ammonium ylides; R3N->B
    "ammonium_ylide": "[#6-]-[NX4H0+](-[#6;!-])(-[#6;!-])-[#6;!-]",
    "imine": "[#6,#1]-C(=[NX2,NH2+,NX3H+;!$(N-[O,SX2,N]);!r3])-[#6,#1]", # excludes azirines, amidines, guanidines, carbodiimides, isoureas
    "ketenimine": "[#6,#1]-C(=C=[NX2,NH2+,NX3H+])-[#6,#1]",
    "iminium": "[CX3;!$([CX3](-[#7]))]=[NX3+;H0;!$([NX3+](-[O-])-[#6,O]);!$([NX3+](-[NX2-]));!$([NX3+]-[C-])]", # excludes nitrones, nitronates, azomethine ylides, azomethine imides
    "enamine": "C=[C;!$(C(-N)-N)]-[NX3,NX4H+,NX4H2+,NH3+;!$(N1~[#6;X3]~[#6;X3]~[O,S]~[#6;X3]~[#6;X3]-1)](-[#6,#1;!$(C=[O,S,N])])-[#6,#1;!$(C=[O,S,N])]", # excludes enediamines, oxazines, thiazines
    "nitrile": "[#6,#1;!$(C=O)]-[CX2]#[NX1H0,NX2H+]", # excludes cyanates, cyanamides, acyl cyanide
    "dicyanomethylidene": "[NX1]#[CX2]-[CX3](=[*])-[CX2]#[NX1]",
    "acyl_cyanide": "O=C-[CX2]#[NX1H0,NX2H+]",
    "azomethine_ylide": "[CX3-]-[NX3+](-[!#1])=[CX3]",
    ## 2x N
    "hydrazine": "[!$(C=[O,N])]-[NX3,NX4+,NX4H2+,NH3+](-[!$(C=[O,N])])-[NX3,NX4+,NX4H2+,NH3+](-[!$(C=[O,N])])-[!$(C=[O,N])]",
    "hydrazone": "[#6,#1]-C(=[NX2,NX3H+;!r5]-[NX3,NX4+,$([NX2]=[P,S])])-[!#7;!#8]", # excludes pyrazolines, amidrazones, N-amino imidates
    "azine": "[CX3]=[NX2,NX3H+]-[NX2,NX3H+]=[CX3]",
    "azine_N-oxide": "[CX3]=[NX3+](-[O-])-[NX2,NX3H+]=[CX3]",
    "azo": "[*]-[NX2,NX3H+;!$([N+]-[O-])]=[NX2,NX3H+;!$([N+]-[O-])]-[*]", # excludes azodicarboxylates
    "diazo": "C=[NX2+]=[NX1-]",
    "diazonium": "[#6]-[NX2+]#[NX1]",
    "azomethine_imide": "[CX3]-,=[NX3+;$(N(=C)(-[N-])),$(N(-[C-])=N)](-[*])-,=[NX2]",
    "amidine": "[#6,#1]-C(=[NX2,NX3H+]-[!O;!N])-[#7X3]([!N])[!N]",
    "amidinium": "[#6,#1]-C(=[NX3H0+]-[!O;!N])-[#7X3]([!N])[!N]",
    "aminal": "[#7]-[CX4;!$(C(N)(N)[O,S,N])]-[#7]",
    "ketene_aminal": "C=C(-[NX3,NX4+])-[NX3,NX4+]",
    "carbodiimide": "[*]-[NX2,NX3H+]=C=[NX2,NX3H+]-[*]",
    "cyanamide": "[#7;!X4]-[CX2]#[NX1H0,NX2H+]",
    ## 3+ N
    "azide": "[!$(C=O)]-[$([NX2]=[NX2+]=[NX1-]),$([NX2-]-[NX2+]#[NX1])]",
    "acyl_azide": "O=C-[$([NX2]=[NX2+]=[NX1-]),$([NX2-]-[NX2+]#[NX1])]",
    "triazene": "N=N-[NX3]",
    "guanidine": "[NX3]-C(=[NX2,NX3H+,NH2+])-[NX3]",
    "guanidinium": "[NX3]-C(=[NX3H0+])-[NX3]",
    "amidrazone": "[#6,#1]-[CX3](-,=N-[NX3])-,=N",
    "orthoamide": "[#6,#1]-C(-[#7])(-[#7])-[#7]",
    "tetraamino_methane": "[#7]-[CX4](-[#7])(-[#7])-[#7]",

    # Oxygen + Nitrogen
    ## 2 hetero atoms
    "cyanohydrin": "O-[CX4]-C#[NX1]",
    "1,2-amino_alcohol": "N-[CX4]-[CX4]-[OH]",
    "hemiaminal": "[#7]-[CX4;!$(C(N)(O)[O,S,N,n])]-[OH]",
    "O,N-acetal": "[#7]-[CX4;!$(C(N)(O)[O,S,N,n])]-O-[!#1]",
    "ketene_O,N-acetal": "C=C(-[#7])-O",
    "hydroxylamine": "[O-,OH,OH2+]-[NX3,NX4H+,NX4H2+,NH3+](-[#6,#1;!$(C=[O,S,N])])-[#6,#1;!$(C=[O,S,N])]",
    "O-organyl_hydroxylamine": "[#6,#14;!$(C=[O,S,N])]-[O;!$(O1-N~[#6;X3]~[#6,#7;X3]~[#6,#7;X3]~[#6;X3]-1)]-[NX3,NX4H+,NX4H2+,NH3+](-[#6,#1;!$(C=[O,S,N])])-[#6,#1;!$(C=[O,S,N])]", # excludes hydroxamate esters, amidines, imidates, oxadiazines
    "aminoxyl_radical": "[OX1H0;!-]-[NX3;!$(N=*)]",
    "oxime": "[#6,#1]-C(=[NX2,NX3H+]-O-[!$(C=O)])-[!#7;!#8]", # excludes amidoximes, N-oxy imidates, oxime-esters
    "nitroso": "[!#7;!#8]-[NX2]=O",
    "nitrone": "[!#8;!#7]-[NX3+]([O-])=C(-[#6,#1])-[#6,#1]",
    "N-oxoammonium": "O=[NX3+](-[#6])-[#6]",
    "N-oxide": "[O-]-[NX4+]",
    "aromatic_N-oxide": "[O-]-[nX3+](:a):a",
    "nitrile_oxide": "[*]-[CX2]#[NX2+]-[O-]",
    "fulminate": "[*]-O-[NX2+]#[CX1-]",
    "cyanate": "[*]-O-C#N",
    "isocyanate": "[*]-[NX2]=C=O",
    "primary_amide": "[#6,#1]-C(=O)-[NX3H2]",
    "secondary_amide": "[#6,#1]-C(=O)-[NX3H]-[#6,#14;!$(C=[O,S])]",
    "tertiary_amide": "[#6,#1]-C(=O)-[#7;X3]([#6,#14;!$(C=[O,S])])[#6,#14;!$(C=[O,S])]", # includes pyrrolide amides
    "acyl_imine": "[#6,#1]-C(=O)-[NX2]=[#6]",
    "imidate": "[#6,#1]-C(=[NX2,NX3H+]-[!#8;!#7])-O-[*]",
    ## 3 hetero atoms
    "diazeniumdiolate": "[*]-[N+]([O-])=[NX2]-[O-]",
    "amide_hemiacetal": "[#6,#1]-C(-[#7])(-[OH])-O-[!#1]",
    "amide_acetal": "[#6,#1]-C(-[#7])(-O-[!#1])-O-[!#1]",
    "amide_O,S-acetal": "[#6,#1]-C(-[#7])(-O)-[SX2]",
    "amide_thioacetal": "[#6,#1]-C(-[#7])(-[SX2])-[SX2]",
    "carboxylic_aminal": "[#6,#1]-C(-[OH])(-[#7])-[#7]",
    "ester_aminal": "[#6,#1]-C(-O)(-[#7])-[#7]",
    "thioester_aminal": "[#6,#1]-C(-[SX2])(-[#7])-[#7]",
    "O-acyl_hydroxylamine": "[*]-C(=O)-O-[NX3,NX4H+,NX4H2+,NH3+](-[#6,#1;!$(C=[O,S,N])])-[#6,#1;!$(C=[O,S,N])]",
    "amidoxime": "[#6,#1]-C(=[NX2,NX3H+]-O)-[NX3]",
    "hydroxamic_acid": "[#6,#1]-C(=O)-[NX3](-[OH,O-,OH2+])-[#6,#14,#1;!$(C=[O,S,N])]",
    "hydroxamate": "[#6,#1]-C(=O)-[NH]-O-[!$(C=O)]",
    "Weinreb_amide": "[#6,#1]-C(=O)-[NX3](-[#6;!$(C=[S,O,N])])-O-[#6;!$(C=[S,O,N])]",
    "hydrazide": "C(=O)-[NX3;!$(N(C=O)C=O)]-[NX4+,#7X3,$([NX2]=[CX3,PX4])]",
    "nitro": "[NX3+](=O)([O-])-[!O;!N;!$([cH0]1[cH0]c(-[N+](=O)[O-])[cH]c(-[N+](=O)[O-])[cH]1);!$([cH0]1[cH]c(-[N+](=O)[O-])[cH0]c(-[N+](=O)[O-])[cH]1);!$([cH0]1[cH][cH][cH][cH][cH0]1-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]);!$([cH0]1[cH][cH][cH0](-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])])[cH][cH]1)]", # excludes nitrate, N-nitro, picryl, nosyls
    "nitrite": "[*]-O-[NX2]=O",
    "nitrosamine": "[#7]-[NX2]=O",
    "imide": "[$(C(=O)-[#6,#1])]-N(-[!O;!#7;!$(C=O)])-[$(C(=O)-[#6,#1])]",
    "urea": "[#7X3,$([NX2]=[CX3,PX4]);!$(N-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[#6])]-[C;!$(C1(=O)NCCN1)](=[O;!$(O=C1[#7]~[#6]C(=O)N1)])-[#7X3,$([NX2]=[CX3,PX4]);!$(N-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[#6])]", # excludes ureido ring, hydantoin, sulfonylurea
    "ureido_ring": "O=C1NCCN1",
    "sulfonylurea": "[#7X3,$([NX2]=[CX3,PX4])]-C(=O)-N-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[#6]",
    "isourea": "[NX3]-C(=[NX2,NX3+])-O-[*]",
    "carbamate": "O-C(=O)-[NX3]",
    "imidocarbonate": "O-C(=[NX2,NX3H+])-O",
    "N-amino_imidate": "[#6,#1]-C(=[NX2,NX3H+;!$([NX2+]=[NX1-])]-[NX3,NX4+])-O-[*]",
    "N-oxy_imidate": "[#6,#1]-C(=[NX2,NX3H+;!$(N1OC=,:COC=,:1)]-O-[*])-O-[*]",
    "nitronate": "[O-,OX2]-[NX3+]([O-,OX2])=C(-[#6,#1])-[#6,#1]",
    "azoxy": "[#6]-[N+]([O-])=[N,NH+;!$([N+]-[O-])]-[#6]",
    "deltic_monoamide": "O=c1c(O)c1([N;!+])",
    "deltamide": "O=c1c([N;!+])c1([N;!+])",
    ## 4+ hetero atoms
    "oxime-ester": "[#6,#1]-C(=[NX2,NX3H+]-O-C(=O))-[!#7;!#8]",
    "orthocarbamate": "[#7]-[CX4](-O)(-O)-O",
    "monothio_orthocarbamate": "[#7]-[CX4](-O)(-O)-[SX2]",
    "dithio_orthocarbamate": "[#7]-[CX4](-O)(-[SX2])-[SX2]",
    "trithio_orthocarbamate": "[#7]-[CX4](-[SX2])(-[SX2])-[SX2]",
    "urea_acetal": "[#7]-[CX4](-[#7])(-O)-O",
    "urea_O,S-acetal": "[#7]-[CX4](-[#7])(-O)-[SX2]",
    "urea_thioacetal": "[#7]-[CX4](-[#7])(-[SX2])-[SX2]",
    "O,N,N,N-carbon": "[#7]-[CX4](-[#7])(-[#7])-O",
    "S,N,N,N-carbon": "[#7]-[CX4](-[#7])(-[#7])-S",
    "O-acyl_hydroxamate": "[#6,#1]-C(=O)-[NX3;!$(N(C=O)C=O)]-O-C(=O)",
    "N-amino_imide": "[#6,#1]-C(=O)-[NX3](-C(=O)-[#6,#1])-[#7]",
    "N-oxy_imide": "[#6,#1]-C(=O)-[NX3](-C(=O)-[#6,#1])-[O;!$(O-C(=O)-[#6,#1])]", # excludes N-acloxy imide
    "N-acyloxy_imide": "[#6,#1]-C(=O)-[NX3;!$(N1C(=O)-C-C-C1(=O))](-C(=O)-[#6,#1])-O-C(=O)-[#6,#1]", # excludes NHS ester
    "NHS_ester": "[#6,#1]-C(=O)-O-N1C(=O)-C-C-C1(=O)",
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
    # the (acyclic) aliphatic side-chains are limited to be connected to: non-carbon atoms, non-sp3 carbons atoms, sp3 carbon atoms in a ring, or quaternary sp3 carbons: [!C,$([C;!X4,R,X4H0])]
    # exception: methyl is allowed to be connected to carbons with exactly one H atom as long as it is not part of: isopropyl (including valine), s-butyl (including isoleucine), isobutyl (including leucine), isoamyl, lactyl, methacroyl, 
    # exception: t-butyl and thexyl are allowed to be connected to carbon atoms with any number of hydrogen atoms
    # [!#1] is used because explicit hydrogens are added to the SMILES prior to counting
    ## saturated
    "methyl": "[CH3]-[!#1;!O;!C,$([C;!X4,R,X4H0;!$(C(=O))]),$([CH;X4;!$([CH]([#1])([CH3])([CH3,$([CH2][CH3])])-[!C,$([C;!X4,R,X4H0]);!#1]);!$([CH]([#1])([CH3])([CH3])[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]);!$([CH]([#1])([CH3])([CH3])[CH2][CH2]-[!C,$([C;!X4,R,X4H0]);!#1]);!$([CH]([#1])(O)C=O);!$([CH]([#1])([CH](N)C=O)[$([CH3]),$([CH2][CH3])]);!$([CH]([#1])([CH3])[CH2][CH](N)C=O)]);!$(C(-[CH3])(-[CH3])(-[CH3,$([CH2]-[CH3]),$([CH]([CH3])[CH3])])[!C,$([C;!X4,R,X4H0]);!#1]);!$(C(-[CH3])(-[CH3])(-[CH3])[CH2][!C,$([C;!X4,R,X4H0]);!#1]);!$(C(-[CH3])(-[CH3])=[CH]-[CH2]);!$(C(-[CH3])(=[CH2])[CH2][CH2][!C,$([C;!X4,R,X4H0]);!#1]);!$(C(=[CH2])([CH3])C=O);!$([cH0]1[cH][cH][cH][cH][cH0]1);!$([cH0]1[cH][cH][cH0][cH][cH]1);!$([cH0]1[cH][cH0][cH][cH][cH]1);!$([cH0]1[cH0]c([CH3])[cH]c([CH3])[cH]1);!$([cH0]1[cH]c([CH3])[cH0]c([CH3])[cH]1);!$([SX4](=O)(=O));!$([SX4+](-[O-])(=O));!$([SX4+2](-[O-])(-[O-]));!$([Si]([CH3])([CH3])[CH3]);!$([Si]([CH3])([CH3])C([CH3])([CH3])[CH3]);!$([Si]([CH3])([CH3])[CH]([CH3])[CH3]);!$([Si]([CH3])([CH3])[cH0]1[cH][cH][cH][cH][cH]1)]",
    ##!! "methyl" excludes: methoxy; methoxymethyl; acetoxy; acetyl; methyls on t-butyl, t-pentyl, neopentyl, thexyl, prenyl; tolyls; mesyl; mesityl; methyls on TMS, TBDMS, etc.;
    "methoxy": "[CH3]-[O;!$(O(-[CH2]-[!C,$([C;!X4,R]);!#1])-[CH3])]-[!$(C=O);!$([cH0]1[cH0][cH][cH][cH][cH]1);!$([cH0]1[cH][cH0][cH][cH][cH]1);!$([cH0]1[cH][cH][cH0][cH][cH]1);!$([cH0]1[cH0]([$(O[CH3]),$([OH])])[cH][cH0][cH][cH]1);!$([cH0]1[cH0]([$(O[CH3]),$([OH])])[cH][cH][cH0][cH]1)]", # excludes methoxymethyl, carbomethoxy, methoxyphenyls
    "methoxymethyl": "[CH3]-O-[CH2]-[!C,$([C;!X4,R]);!#1]",
    "ethyl": "[CH3]-[CH2]-[!C,$([C;!X4,R,X4H0;!$(C(-[CH3])(-[CH3]))]);!$([Si]([CH2][CH3])([CH2][CH3])[CH2][CH3]);!O;!#1]", # excludes ethoxy, ethyls on TES and the ethyl on t-pentyl
    "ethylene": "[!#1;!C,$([C;!X4,R,H0])]-[CH2]-[CH2]-[!#1;!C,$([C;!X4,R,H0])]",
    "ethoxy": "[CH3]-[CH2]-O-[!$(C=O)]", # excludes carboethoxy
    "n-propyl": "[CH3]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!O;!#1]", # excludes propoxy
    "propylene": "[!#1;!C,$([C;!X4,R,H0])]-[CH2]-[CH2]-[CH2]-[!#1;!C,$([C;!X4,R,H0])]",
    "n-propoxy": "[CH3]-[CH2]-[CH2]-O-[*]",
    "isopropyl": "[CH3]-[CH](-[CH3])-[!C,$([C;!X4,R,X4H0]);!O;!#1;!$(C(-[CH3])-[CH3]);!$([Si]([CH]([CH3])[CH3])([CH]([CH3])[CH3])([CH]([CH3])[CH3]));!$([Si]([CH]([CH3])[CH3])([CH3])([CH3]));!$([cH0]1[cH0]c(-[CH]([CH3])[CH3])[cH]c(-[$([CH]([CH3])[CH3]),#1])[cH]1);!$([cH0]1[cH]c(-[CH]([CH3])[CH3])[cH0]c(-[CH]([CH3])[CH3])[cH]1)]", # excludes the isopropyl on thexyl; isopropoxy; 2,6-di and 2,4,6-triisopropylphenyl; isopropyls on common silyls
    "isopropoxy": "[CH3]-[CH](-[CH3])-O-[*]",
    "n-butyl": "[CH3]-[CH2]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]",
    "butylene": "[!#1;!C,$([C;!X4,R,H0])]-[CH2]-[CH2]-[CH2]-[CH2]-[!#1;!C,$([C;!X4,R,H0])]",
    "isobutyl": "[CH3]-[CH](-[CH3])-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]",
    "s-butyl": "[CH3]-[CH2]-[CH](-[CH3])-[!C,$([C;!X4,R,X4H0]);!#1]",
    "t-butyl": "[CH3]-C(-[CH3])(-[CH3])-[!O;!#1;!$([CH2]([!C,$([C;!X4,R,X4H0])])C([CH3])([CH3])[CH3]);!$([Si]([CH3])([CH3])C([CH3])([CH3])[CH3]);!$([Si]([cH0]1[cH][cH][cH][cH][cH]1)([cH0]1[cH][cH][cH][cH][cH]1)C([CH3])([CH3])[CH3]);!$([cH0]1[cH0]c(-C([CH3])([CH3])[CH3])[cH]c(-[$(C([CH3])([CH3])[CH3]),#1])[cH]1);!$([cH0]1[cH]c(-C([CH3])([CH3])[CH3])[cH0]c(-C([CH3])([CH3])[CH3])[cH]1)]", # excludes neopentyl, t-butoxy, t-Boc, t-butyls on TBDMS and TBDPS, 2,6-di and 2,4,6-tri-tert-butylphenyl
    "t-butoxy": "[CH3]-C(-[CH3])(-[CH3])-O-[!$(C(=O)(O)[!#6])]", # excludes tBoc
    "n-pentyl": "[CH3]-[CH2]-[CH2]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]",
    "pentylene": "[!#1;!C,$([C;!X4,R,H0])]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!#1;!C,$([C;!X4,R,H0])]",
    "t-pentyl": "[CH3]-[CH2]-C(-[CH3])(-[CH3])-[!C,$([C;!X4,R,X4H0]);!#1]",
    "isoamyl": "[CH3]-[CH](-[CH3])-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]",
    "neopentyl": "[CH3]-C(-[CH3])(-[CH3])-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]",
    "n-hexyl": "[CH3]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]",
    "hexylene": "[!#1;!C,$([C;!X4,R,H0])]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!#1;!C,$([C;!X4,R,H0])]",
    "thexyl": "[CH3]-[CH](-[CH3])-C(-[CH3])(-[CH3])-[!#1]",
    "n-heptyl":  "[CH3]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]",
    "heptylene": "[!#1;!C,$([C;!X4,R,H0])]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!#1;!C,$([C;!X4,R,H0])]",
    "n-octyl":  "[CH3]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]",
    "octylene": "[!#1;!C,$([C;!X4,R,H0])]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!#1;!C,$([C;!X4,R,H0])]",
    "n-nonyl":  "[CH3]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]",
    "nonylene": "[!#1;!C,$([C;!X4,R,H0])]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!#1;!C,$([C;!X4,R,H0])]",
    "n-decyl":  "[CH3]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]",
    "decylene": "[!#1;!C,$([C;!X4,R,H0])]-[CH2]--[CH2][CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!#1;!C,$([C;!X4,R,H0])]",
    "lauryl": "[CH3]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]", # dodecyl
    "cetyl":  "[CH3]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]", # hexadecyl
    "stearyl": "[CH3]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[!C,$([C;!X4,R,X4H0]);!#1]", # octadecyl
    # cyclic
    "cyclopropyl": "[!#1]-[CX4;H1]1-[CH2]-[CH2]-1",
    "cyclopropylidene": "[!O;!S]=[CX3]1-[CX4]-[CX4]-1",
    "1,1-cyclopropandiyl": "[*]-[CH0](-[*])1-[CH2]-[CH2]-1",
    "cyclobutyl":  "[!#1]-[CX4;H1]1-[CH2]-[CH2]-[CH2]-1",
    "cyclopentyl":  "[!#1]-[CX4;H1]1-[CH2]-[CH2]-[CH2]-[CH2]-1",
    "cyclohexyl":  "[!#1]-[CX4;H1]1-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-1",
    "1-adamantyl": "[!#1]-[CX4;H0]1(-[CH2]2)-[CH2]-[CH](-[CH2]3)-[CH2]-[CH](-[CH2]-1)-[CH2]-[CH]-2-3",
    "2-adamantyl": "[CH]1(-[CH2]2)-[CX4;H1](-[!#1])-[CH](-[CH2]3)-[CH2]-[CH](-[CH2]-1)-[CH2]-[CH]-2-3",
    ## unsaturated
    "vinyl": "[CH2]=[CH]-[!$([CH2]);!$(C=O);!#1]", # excludes acryloyl
    "methylidene": "[CH2;!$([CH2]=C=[*]);!$([CH2]=[CH]);!$([CH2]=C([CH3])C=O);!$([CH2]=C(-[CH3])[CH2][CH2][!C,$([C;!X4,R,X4H0]);!#1])]=[*]", # excludes vinyl, methacryloyl, isopentenyl, vinylidene
    "vinylidene": "[CH2]=C=[*]",
    "allyl": "[CH2]=[CH]-[CH2]-[!C,$([C;R]);!#1]",
    "propargyl": "[CH]#C-[CH2]-[!C,$([C;R]);!#1]",
    # acyls
    "formyl": "[#1]-C(=O)-[!O;!#6]",
    "formoxy": "[#1]-C(=O)-O",
    "acetyl": "[!O;!C]-C(=O)-[CH3]", # excludes acetoxy, acetoacetyl
    "acetoxy": "[CH3]-C(=O)-O-[!#1]",
    "glycolyl": "[!C;!#1]-C(=O)-[CH2]-O",
    "lactyl": "[!C;!#1]-C(=O)-[CH](O)-[CH3]",
    "acryloyl": "[!C;!#1]-C(=O)-[CH]=[CH2]",
    "methacryloyl": "[!C;!#1]-C(=O)-C([CH3])=[CH2]",
    "carbomethoxy": "[CH3]-O-C(=O)-[!#1]",
    "carboethoxy": "[CH3]-[CH2]-O-C(=O)-[!#1]",
    "t-Boc": "[CH3]-C(-[CH3])(-[CH3])-O-C(=O)-[!#1;!#6]",
    "Fmoc": "[cH0]12[cH][cH][cH][cH][cH0]2-[cH0]2[cH][cH][cH][cH][cH0]2-[CH]1-[CH2]-O-C(=O)-[!#1;!#6]",
    # common diacyls
    "oxalyl": "[!C;!#1]~[#6](=O)-,:[#6](=O)~[!C;!#1]",
    "pyruvyl": "[CH3]-C(=O)-C(=O)-[!#6;!#1]",
    "malonyl": "[!#6;!#1]-C(=O)-C-C(=O)-[!#6;!#1]",
    "acetoacetyl": "[CH3]-C(=O)-C-C(=O)-[!#1;!$([CH3])]",
    "acetylacetone": "[CH3]-C(=O)-C-C(=O)-[CH3]",
    "succinyl": "[$(C(=O)-[!#6;!#1])]-[C;!$(C-N);!$(C(~[#6])(~[#6])~[#6])]-[C;!$(C-N);!$(C(~[#6])(~[#6])~[#6])]-[$(C(=O)-[!#6;!#1])]", # excludes aspartate
    "glutaryl": "[$(C(=O)-[!#6;!#1])]-[C;!$(C-N);!$(C(~[#6])(~[#6])~[#6])]-[C;!$(C(~[#6])(~[#6])~[#6])]-[C;!$(C-N);!$(C(~[#6])(~[#6])~[#6])]-[$(C(=O)-[!#6;!#1])]", # excludes glutamate
    "adipoyl": "[$(C(=O)-[!#6;!#1])]-[C;!$(C(~[#6])(~[#6])~[#6])]-[C;!$(C(~[#6])(~[#6])~[#6])]-[C;!$(C(~[#6])(~[#6])~[#6])]-[C;!$(C(~[#6])(~[#6])~[#6])]-[$(C(=O)-[!#6;!#1])]",
    "maleoyl_acyclic": r"[!#6;!#1]-C(=O)\C=C/C(=O)-[!#6;!#1]",
    "maleoyl_cyclic": r"[$([#6](=O)[!#6;!#1])]@[#6X3;!$(c1ccccc1)]@[#6X3;!$(c1ccccc1)]@[$([#6](=O)[!#6;!#1])]",
    "fumaroyl": "[$(C(=O)-[!#6;!#1])]/C=C/[$(C(=O)-[!#6;!#1])]",
    "acetylenedicarboxoyl": "[!#6;!#1]-C(=O)-C#C-C(=O)-[!#6;!#1]",
    "phthaloyl": "[$([#6](=O)[!#6;!#1])]c1ccccc1[$([#6](=O)[!#6;!#1])]",
    "isophthaloyl": "[!#6;!#1]-C(=O)-c1cc(-C(=O)-[!#6;!#1])ccc1",
    "terephthaloyl": "[!#6;!#1]-C(=O)-c1ccc(-C(=O)-[!#6;!#1])cc1",
    # aromatic
    "phenyl": "[!O;!$(C=O);!C,$([C;!H2]),$([CH2]-[C;!$(C@*)]);!$([CH]=[CH]-[CH2,$(C=O)]);!$([Si](-[cH0]1[cH][cH][cH][cH][cH]1)([CH3])[CH3]);!$([Si](-[cH0]1[cH][cH][cH][cH][cH]1)(-[cH0]1[cH][cH][cH][cH][cH]1)-[$([cH0]1[cH][cH][cH][cH][cH]1),$(C([CH3])([CH3])[CH3])])]-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1", # excludes benzyl, phenoxy, cinnamyl, cinnamoyl
    "phenoxy": "[*]-O-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1",
    "benzyl": "[!C,$([C;!X4,R,X4H0]);!#1;!O]-[CH2]-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1", # excludes benzoxy
    "benzoxy": "[*]-O-[CH2]-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1",
    "benzoyl": "[!O]-C(=O)-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1",
    "benzoate": "[*]-O-C(=O)-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1",
    "o-tolyl": "[!$([$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])])]-[cH0]1:[cH0](-[CH3]):[cH]:[cH]:[cH]:[cH]:1", # excludes o-tosyl
    "m-tolyl": "[*]-[cH0]1:[cH]:[cH0](-[CH3]):[cH]:[cH]:[cH]:1",
    "p-tolyl": "[!$([$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])])]-[cH0]1:[cH]:[cH]:[cH0](-[CH3]):[cH]:[cH]:1", # excludes p-tosyl
    "2-pyridyl": "[*]-[cH0]1:n:[cH]:[cH]:[cH]:[cH]:1",
    "3-pyridyl": "[*]-[cH0]1:[cH]:n:[cH]:[cH]:[cH]:1",
    "4-pyridyl": "[*]-[cH0]1:[cH]:[cH]:n:[cH]:[cH]:1",
    "cinnamyl": "[!#1]-[CH2]-[CH]=[CH]-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1",
    "cinnamoyl": "[*]-C(=O)-[CH]=[CH]-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1",
    "mesityl": "[*]-[cH0]1:[cH0](-[CH3]):[cH]:[cH0](-[CH3]):[cH]:[cH0](-[CH3]):1",
    "2,6-diisopropylphenyl": "[*]-[cH0]1:[cH0](-[CH](-[CH3])-[CH3]):[cH]:[cH]:[cH]:[cH0](-[CH](-[CH3])-[CH3]):1",
    "2,4,6-triisopropylphenyl": "[*]-[cH0]1:[cH0](-[CH](-[CH3])-[CH3]):[cH]:[cH0](-[CH](-[CH3])-[CH3]):[cH]:[cH0](-[CH](-[CH3])-[CH3]):1",
    "2,6-di-tert-butylphenyl": "[*]-[cH0]1:[cH0](-C(-[CH3])(-[CH3])-[CH3]):[cH]:[cH]:[cH]:[cH0](-C(-[CH3])(-[CH3])-[CH3]):1",
    "2,4,6-tri-tert-butylphenyl": "[*]-[cH0]1:[cH0](-C(-[CH3])(-[CH3])-[CH3]):[cH]:[cH0](-C(-[CH3])(-[CH3])-[CH3]):[cH]:[cH0](-C(-[CH3])(-[CH3])-[CH3]):1",
    "picryl": "[*]-[cH0]1:[cH0](-[N+](=O)-[O-]):[cH]:[cH0](-[N+](=O)-[O-]):[cH]:[cH0](-[N+](=O)-[O-]):1",
    "trityl": "[*]-[CX4H0](-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1)(-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1)(-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH]:1)",
    # hydroxyphenyls and related
    ## mono-sub
    "o-hydroxyphenyl": "[!#1]-[cH0]1:[cH0](-[OH]):[cH]:[cH]:[cH]:[cH]:1",
    "m-hydroxyphenyl": "[!#1]-[cH0]1:[cH]:[cH0](-[OH]):[cH]:[cH]:[cH]:1",
    "p-hydroxyphenyl": "[!$([CH]=[CH]-C=O)]-[cH0]1:[cH]:[cH]:[cH0](-[OH]):[cH]:[cH]:1", # excludes coumaroyl
    "coumaroyl":"[*]-C(=O)-[CH]=[CH]-[cH0]1:[cH]:[cH]:[cH0](-O):[cH]:[cH]:1",
    "o-methoxyphenyl": "[!#1]-[cH0]1:[cH0](-O-[CH3]):[cH]:[cH]:[cH]:[cH]:1",
    "m-methoxyphenyl": "[!#1]-[cH0]1:[cH]:[cH0](-O-[CH3]):[cH]:[cH]:[cH]:1",
    "p-methoxyphenyl": "[!#1]-[cH0]1:[cH]:[cH]:[cH0](-O-[CH3]):[cH]:[cH]:1",
    ## di-sub
    "2,3-dihydroxyphenyl": "[!#1]-[cH0]1:[cH0](-[OH]):[cH0](-[OH]):[cH]:[cH]:[cH]:1",
    "2,4-dihydroxyphenyl": "[!#1]-[cH0]1:[cH0](-[OH]):[cH]:[cH0](-[OH]):[cH]:[cH]:1",
    "2,5-dihydroxyphenyl": "[!#1]-[cH0]1:[cH0](-[OH]):[cH]:[cH]:[cH0](-[OH]):[cH]:1",
    "2,6-dihydroxyphenyl": "[!#1]-[cH0]1:[cH0](-[OH]):[cH]:[cH]:[cH]:[cH0](-[OH]):1",
    "3,4-dihydroxyphenyl": "[!#1]-[cH0]1:[cH]:[cH0](-[OH]):[cH0](-[OH]):[cH]:[cH]:1",
    "3,5-dihydroxyphenyl": "[!#1]-[cH0]1:[cH]:[cH0](-[OH]):[cH]:[cH0](-[OH]):[cH]:1",
    "3-methoxy-4-hydroxyphenyl": "[!$([CH2]);!$(C=O);!$([CH]=[CH]-C=O)]-[cH0]1:[cH]:[cH0](-O-[CH3]):[cH0](-[OH]):[cH]:[cH]:1", # excludes vanillyl, vanilloyl, feruloyl
    "3-hydroxy-4-methoxyphenyl": "[!$([CH2]);!$(C=O)]-[cH0]1:[cH]:[cH0](-[OH]):[cH0](-O-[CH3]):[cH]:[cH]:1", # excludes isovanillyl, isovanilloyl
    "3,4-dimethoxyphenyl": "[!$([CH2]);!$(C=O)]-[cH0]1:[cH]:[cH0](-O-[CH3]):[cH0](-O-[CH3]):[cH]:[cH]:1", # excludes veratryl, veratroyl
    "vanillyl": "[!#1]-[CH2]-[cH0]1:[cH]:[cH0](-O-[CH3]):[cH0](-[OH]):[cH]:[cH]:1",
    "isovanillyl": "[!#1]-[CH2]-[cH0]1:[cH]:[cH0](-[OH]):[cH0](-O-[CH3]):[cH]:[cH]:1",
    "vanilloyl": "[*]-C(=O)-[cH0]1:[cH]:[cH0](-O-[CH3]):[cH0](-[OH]):[cH]:[cH]:1",
    "isovanilloyl": "[*]-C(=O)-[cH0]1:[cH]:[cH0](-[OH]):[cH0](-O-[CH3]):[cH]:[cH]:1",
    "feruloyl": "[*]-C(=O)-[CH]=[CH]-[cH0]1:[cH]:[cH0](-O-[CH3]):[cH0](-[OH]):[cH]:[cH]:1",
    "veratryl": "[!#1]-[CH2]-[cH0]1:[cH]:[cH0](-O-[CH3]):[cH0](-O-[CH3]):[cH]:[cH]:1",
    "veratroyl": "[*]-C(=O)-[cH0]1:[cH]:[cH0](-O-[CH3]):[cH0](-O-[CH3]):[cH]:[cH]:1",
    ### tri-sub
    "pyrogallol": "O-[cH0]1:[cH0](-O):[cH0](-O):[cH]:[cH]:[cH]:1", 
    "3,4,5-trioxyphenyl": "O-[cH0]1:[cH0](-O):[cH0](-O):[cH]:[cH0;!$([cH0]1(-[$(C=O),$([CH2])])[cH][cH0](-O[CH3])[cH0](-[OH])[cH0](-O[CH3])[cH]1)](-[!#1;!$(C=O)]):[cH]:1", # excludes galloyl, syringyl, syringoyl
    "phloroglucinol": "O-[cH0]1:[cH]:[cH0](-O):[cH]:[cH0](-O):[cH]:1", 
    "2,4,6-trioxyphenyl": "O-[cH0]1:[cH0](-[!#1;!$(C=O)]):[cH0](-O):[cH]:[cH0](-O):[cH]:1", 
    "galloyl": "[*]-C(=O)-[cH0]1:[cH]:[cH0](-O):[cH0](-O):[cH0](-O):[cH]:1",
    "syringyl": "[!#1]-[CH2]-[cH0]1:[cH]:[cH0](-O-[CH3]):[cH0](-[OH]):[cH0](-O-[CH3]):[cH]:1",
    "syringoyl": "[!#1]-C(=O)-[cH0]1:[cH]:[cH0](-O-[CH3]):[cH0](-[OH]):[cH0](-O-[CH3]):[cH]:1",
}

MAIN_GROUP: Dict[str, Dict] = {
    # Organometallics
    "Organometallics" : {
        "organo_lithium": "[#6]-[Li]",
        "Grignard": "[#6]-[Mg]-[Cl,Br,IX1]",
        "organo_aluminum": "[#6]-[Al]",
        "organo_mercury": "[#6]-[Hg]",
        "organo_zinc": "[#6]-[Zn]",
        "stannane": "[#6]-[SnX4](-[#6])(-[#6])-[#6]",
        "stannyl_monohydride": "[#1]-[SnX4](-[#6])(-[#6])-[#6]",
        "stannyl_monohalide": "[F,Cl,Br,IX1]-[SnX4](-[#6])(-[#6])-[#1]",
        "tributyl_stannyl": "[*]-[SnX4](-[CH2]-[CH2]-[CH2]-[CH3])(-[CH2]-[CH2]-[CH2]-[CH3])-[CH2]-[CH2]-[CH2]-[CH3]",
    },
    # Boron
    "Boron": {
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
        "borate": "[BX4-;H0]",
        "borolate": "[BX4-]1-[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-1",
        ## B-O
        "boranate": "[#6]-[BX4-](-O)(-[#6])[#6]",
        "borinic_acid": "[#6,#1]-[BX3](-[OH,OH2+])-[#6,#1]",
        "borinic_ester": "[#6,#1]-[BX3;!$([#5]1-,:[#8]-,:[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-,:1)](-O-[!#1])-[#6,#1]", # excludes 1,2-oxaborine
        "borinate": "[#6]-[BX4-](-O)(-O)-[#6]",
        "boronic_acid": "[#6,#1]-[BX3](-[OH,OH2+])-[OH,OH2+]",
        "boronic_mono_ester": "[#6,#1]-[BX3;!$([#5]1-,:[#8]-,:[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-,:1)](-[O-,OH,OH2+])-O-[!#1]", # excludes 1,2-oxaborine
        "boronic_di_ester": "[#6,#1]-[BX3;!$(B1OBOBO1);!$([#5]1-,:[#8]-,:[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-,:1)](-O-[!#1])-O-[!#1]", # excludes boroxine, 1,2-oxaborine
        "boronate": "[#6]-[BX4-](-O)(-O)-O",
        "catecholborane": "[*]-[BX3]1-O-c2ccccc2-O-1",
        "pinnacolborane": "[*]-[BX3]1-O-C([CH3])([CH3])-C1([CH3])[CH3]",
        "orthoboric_mono_ester": "[!#1]-O-[BX3](-[O-,OH,OH2+])-[O-,OH,OH2+]",
        "orthoboric_di_ester": "[!#1]-O-[BX3](-[O-,OH,OH2+])-O-[!#1]",
        "orthoboric_tri_ester": "[!#1]-O-[BX3](-O-[!#1])-O-[!#1]",
        "orthoboric_monoamide": "O-[BX3](-O)-[#7]",
        "orthoboric_diamide": "O-[BX3](-[#7])-[#7]",
        "triamino_borane": "[#7]-[BX3](-[#7])-[#7]",
        "boroxine": "B1-O-B-O-B-O1",
        "1,2-oxaborine": "[#5]1-,:[#8]-,:[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-,:1",
        ## B-N
        "borinic_amide": "[#6,#1]-[BX3,BX4-;!$([#5]1-,:[#7]-,:[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-,:1)](-[#7])-[#6,#1]",
        "boronic_monoamide": "[#6,#1]-[BX3,BX4-;!$([#5]1-,:[#7]-,:[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-,:1);!$([#5]1-,:[#8]-,:[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-,:1);!$([BX4-](-O)(-O)(-[#7]))](-[#7])-O", # excludes 1,2-azaborine, 1,2-oxaborine, boronamidate
        "boronamide": "[#6,#1]-[BX3,BX4-;!$([#5]1-,:[#7]-,:[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-,:1)](-[#7])-[#7]",
        "boronamidate": "[#6,#1]-[BX4-](-O)(-O)-[NX4+,#7X3+]",
        "BODIPY": "[BX4-]1-[#7;X3]2~[#6;X3]~[#6;X3]~[#6;X3]~[#6;X3]~2~[#6;X3]~[#6;X3]3~[#6;X3]~[#6;X3]~[#6;X3]~[#7;X3]~3-1",
        "trispyrazolylborate": "[BX4-](n1cccn1)(n2cccn2)n3cccn3",
        "borazine": "[bX3-]1[nX3+][bX3-][nX3+][bX3-][nX3+]1",
        "carborazine": "[bX3-]1[nX3+]c[nX3+][bX3-]c1",
        "1,2-azaborine": "[#5]1-,:[#7]-,:[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-,:1",
        "1,3-azaborine": "[#5]1-,:[#6;X3]-,:[#7]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-,:1",
        "1,4-azaborine": "[#5]1-,:[#6;X3]-,:[#6;X3]=,:[#7]-,:[#6;X3]=,:[#6;X3]-,:1",
        "oxazaborolidine": "B1NCCO1",
        "benzoxaborole": "B1OCc2ccccc21",
        ## B-F
        "difluoroborate": "[*]-[BX4-](F)(F)-[*]",
        "trifluoroborate": "[*]-[BX4-](F)(F)F",
    },
    # Silicon
    "Silicon": {
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
    },
    # Phosphorus
    "Phosphorus": {
        ## P(III), P(II)
        ### PR3, PR4+
        "phosphine": "[PX3](-[#6,#14,#1])(-[#6,#14,#1])-[#6,#14,#1]",
        "phosphole": "[pX3]1:c:c:c:c:1",
        "phosphinine": "[pX2]1:c:c:c:c:c:1",
        "phosphinine_oxide": "[O-]-[pX3+]1:c:c:c:c:c:1",
        "phosphonium": "[PX4+;!$([P+]-[O-]);!$([P+]-[C-])]",
        "diphosphine": "[PX3]-[PX3]",
        "phosphaalkyne": "[#6]-[CX2]#[PX1]",
        "phosphino_borane": "[BX4-]-[PX4+]",
        ### PR2X
        "phosphinite": "[PX3](-[#6,#1])(-[#6,#1])-O",
        "thiophosphinite": "[PX3](-[#6,#1])(-[#6,#1])-[SX2]-[#6]",
        "aminophosphine": "[PX3](-[#6,#1])(-[#6,#1])-[#7X3,$([NX2]=[CX3,PX4])]",
        "halophosphine": "[PX3](-[#6,#1])(-[#6,#1])-[F,Cl,Br,IX1]",
        ### PRX2
        "phosphonite": "[PX3](-[#6,#1])(-O)-O",
        "thiophosphonite": "[PX3](-[#6,#1])(-[SX2])-O",
        "dithiophosphonite": "[PX3](-[#6,#1])(-[SX2])-[SX2]",
        "phosphonamidite": "[PX3](-[#6,#1])(-O)-[#7X3,$([NX2]=[CX3,PX4])]",
        "diaminophosphine": "[PX3](-[#6,#1])(-[#7X3,$([NX2]=[CX3,PX4])])-[#7X3,$([NX2]=[CX3,PX4])]",
        "halophosphonite": "[PX3](-[#6,#1])(-O)-[F,Cl,Br,IX1]",
        "halothiophosphonite": "[PX3](-[#6,#1])(-[SX2]-[#6])-[F,Cl,Br,IX1]",
        "haloaminophosphine": "[PX3](-[#6,#1])(-[#7X3,$([NX2]=[CX3,PX4])])-[F,Cl,Br,IX1]",
        "dihalophosphine": "[PX3](-[#6,#1])(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
        ### PX3
        "phosphite": "[PX3](-O)(-O)-O",
        "thiophosphite": "[PX3](-O)(-O)-[SX2]",
        "dithiophosphite": "[PX3](-O)(-[SX2])-[SX2]",
        "trithiophosphite": "[PX3](-[SX2]-[#6])(-[SX2]-[#6])-[SX2]-[#6]",
        "phosphoramidite": "[PX3](-O)(-O)-[#7X3,$([NX2]=[CX3,PX4])]",
        "phosphorodiamidite": "[PX3](-O)(-[#7X3,$([NX2]=[CX3,PX4])])-[#7X3,$([NX2]=[CX3,PX4])]",
        "triaminophosphine": "[PX3](-[#7X3,$([NX2]=[CX3,PX4])])(-[#7X3,$([NX2]=[CX3,PX4])])-[#7X3,$([NX2]=[CX3,PX4])]",
        "thiophosphoramidite": "[PX3](-O)(-[SX2])-[#7X3,$([NX2]=[CX3,PX4])]",
        "dithiophosphoramidite": "[PX3](-[SX2])(-[SX2])-[#7X3,$([NX2]=[CX3,PX4])]",
        "thiophosphorodiamidite": "[PX3](-[SX2])(-[#7X3,$([NX2]=[CX3,PX4])])-[#7X3,$([NX2]=[CX3,PX4])]",
        "phosphoro_halidite": "[PX3](-O)(-O)-[F,Cl,Br,IX1]",
        "thiophosphoro_halidite": "[PX3](-O)(-[SX2])-[F,Cl,Br,IX1]",
        "dithiophosphoro_halidite": "[PX3](-[SX2])(-[SX2])-[F,Cl,Br,IX1]",
        "halophosphoramidite": "[PX3](-O)(-[#7X3,$([NX2]=[CX3,PX4])])-[F,Cl,Br,IX1]",
        "halodiaminophosphine": "[PX3](-[#7X3,$([NX2]=[CX3,PX4])])(-[#7X3,$([NX2]=[CX3,PX4])])-[F,Cl,Br,IX1]",
        "halothiophosphoramidite": "[PX3](-[#7X3,$([NX2]=[CX3,PX4])])(-[SX2])-[F,Cl,Br,IX1]",
        "phosphoro_dihalidite": "[PX3](-O)(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
        "thiophosphoro_dihalidite": "[PX3](-[SX2])(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
        "dihaloaminophosphine": "[PX3](-[#7X3,$([NX2]=[CX3,PX4])])(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
                        
        ## C=P(VI)
        "phosphonium_ylide": "[$([PX4+]-[C-;X3,X2]),$([PX4]=[C;X3,X2])]",
        "carbodiphosphorane": "[$([PX4]=C=[PX4]),$([PX4]=[CX2-]-[PX4+]),$([PX4+]-[CX2-2]-[PX4+])]",
        ## O=P(VI)
        ### O=PR3
        "phosphine_oxide": "[$([PX4]=O),$([PX4+]-[O-])](-[#6,#1])(-[#6,#1])-[#6,#1]",
        ### O=PR2X
        "phosphinic_acid": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-[#6])-[OH,O-,OH2+]",
        "phosphinate": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-[#6])-O-[!#1]",
        "thiophosphinate_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-[#6])-[SX2]-[#6]",
        "phosphinamide": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-[#6])-[#7X3,$([NX2]=[CX3,PX4])]",
        "phosphinic_halide": "[$([PX4]=O),$([PX4+]-[O-])](-[#6,#1])(-[#6,#1])-[F,Cl,Br,IX1]",
        ### O=PRX2
        "phosphonic_acid": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-[OH,O-,OH2+])-[OH,O-,OH2+]",
        "phosphonate_mono_substituted": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-[OH,O-,OH2+])-O-[!#1]",
        "phosphonate_di_substituted": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-O-[!#1])-O-[!#1]",
        "thiophosphonate_mono_ester_mono_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-O-[!#1])-[SX2]-[#6]",
        "dithiophosphonate_di_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-[SX2]-[#6])-[SX2]-[#6]",
        "phosphonamidate": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-O)-[#7X3,$([NX2]=[CX3,PX4])]",
        "thiophosphonamidate_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-[SX2]-[#6])-[#7X3,$([NX2]=[CX3,PX4])]",
        "phosphondiamide": "[$([PX4]=O),$([PX4+]-[O-])](-[#6])(-[#7X3,$([NX2]=[CX3,PX4])])-[#7X3,$([NX2]=[CX3,PX4])]",
        "phosphono_halidate": "[$([PX4]=O),$([PX4+]-[O-])](-[#6,#1])(-O)-[F,Cl,Br,IX1]",
        "phosphonamidic_halide": "[$([PX4]=O),$([PX4+]-[O-])](-[#6,#1])(-[#7X3,$([NX2]=[CX3,PX4])])-[F,Cl,Br,IX1]",
        "phosphonic_dihalide": "[$([PX4]=O),$([PX4+]-[O-])](-[#6,#1])(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
        ### O=PX3
        "phosphate_mono_ester": "[$([PX4]=O),$([PX4+]-[O-])](-[OH,O-,OH2+])(-[OH,O-,OH2+])-O-[!#1;!P]",
        "phosphate_di_ester": "[$([PX4]=O),$([PX4+]-[O-])](-[OH,O-,OH2+])(-O-[!#1;!P])-O-[!#1;!P]",
        "phosphate_tri_ester": "[$([PX4]=O),$([PX4+]-[O-])](-O-[!#1])(-O-[!#1;!P])-O-[!#1;!P]",
        "thiophosphate_mono_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-[OH,O-,OH2+])(-[OH,O-,OH2+])-[SX2]-[!#1]",
        "thiophosphate_di_ester": "[$([PX4](=O)(-[SX2H,SX1-])),$([PX4](-[OH,O-,OH2+])(=S)),$([PX4+](-[OH,O-,OH2+])(-[SX1-])),$([PX4+](-[O-])(-[SX2H]))](-O-[!#1])-O-[!#1]",
        "thiophosphate_mono_ester_mono_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-[OH,O-,OH2+])(-O-[!#1])-[SX2]-[#6]",
        "thiophosphate_di_ester_mono_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-O-[!#1])(-O-[!#1])-[SX2]-[#6]",
        "dithiophosphate_mono_ester_di_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-O-[!#1])(-[SX2]-[#6])-[SX2]-[#6]",
        "trithiophosphate_tri_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-[SX2]-[#6])(-[SX2]-[#6])-[SX2]-[#6]",
        "phosphoramidate_mono_ester": "[$([PX4]=O),$([PX4+]-[O-])](-[OH,O-,OH2+])(-O-[!#1])-[#7X3,$([NX2]=[CX3,PX4])]",
        "phosphoramidate_di_ester": "[$([PX4]=O),$([PX4+]-[O-])](-O-[!#1])(-O-[!#1])-[#7X3,$([NX2]=[CX3,PX4])]",
        "thiophosphoramidate_mono_ester_mono_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-O-[!#1])(-[SX2]-[#6])-[#7X3,$([NX2]=[CX3,PX4])]",
        "dithiophosphoramidate_di_thioester": "[$([PX4]=O),$([PX4+]-[O-])](-[SX2]-[#6])(-[SX2]-[#6])-[#7X3,$([NX2]=[CX3,PX4])]",
        "phosphorodiamidate": "[$([PX4]=O),$([PX4+]-[O-])](-[O-,OX2])(-[#7X3,$([NX2]=[CX3,PX4])])-[#7X3,$([NX2]=[CX3,PX4])]",
        "phosphoramide": "[$([PX4]=O),$([PX4+]-[O-])](-[#7X3,$([NX2]=[CX3,PX4])])(-[#7X3,$([NX2]=[CX3,PX4])])-[#7X3,$([NX2]=[CX3,PX4])]",
        "halophosphate": "[$([PX4]=O),$([PX4+]-[O-])](-[O-,OX2])(-[O-,OX2])-[F,Cl,Br,IX1]",
        "phosphoramidic_halidate": "[$([PX4]=O),$([PX4+]-[O-])](-[#7X3,$([NX2]=[CX3,PX4])])(-[O-,OX2])-[F,Cl,Br,IX1]",
        "phosphorodiamidic_halide": "[$([PX4]=O),$([PX4+]-[O-])](-[#7X3,$([NX2]=[CX3,PX4])])(-[#7X3,$([NX2]=[CX3,PX4])])-[F,Cl,Br,IX1]",
        "dihalophosphate": "[$([PX4]=O),$([PX4+]-[O-])](-[O-,OX2])(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
        "phosphoramidic_dihalide": "[$([PX4]=O),$([PX4+]-[O-])](-[#7X3,$([NX2]=[CX3,PX4])])(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
        "diphosphate": "[$([PX4]=O),$([PX4+]-[O-])](-[O-,OX2])(-[O-,OX2])-O-[$([PX4]=O),$([PX4+]-[O-])](-[O-,OX2])(-[O-,OX2])",
        "triphosphate": "[$([PX4]=O),$([PX4+]-[O-])](-[O-,OX2])(-[O-,OX2])-O-[$([PX4]=O),$([PX4+]-[O-])](-[O-,OX2])-O-[$([PX4]=O),$([PX4+]-[O-])](-[O-,OX2])(-[O-,OX2])",
        
        ## S=P(VI)
        ### S=PR3
        "phosphine_sulfide": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6,#1])(-[#6,#1])-[#6,#1]",
        ### S=PR2X
        "thiophosphinate_ester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-[#6])-O-[!#1]",
        "dithiophosphinate_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-[#6])-[SX2]-[#6]",
        "thiophosphinamide": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-[#6])-[#7X3,$([NX2]=[CX3,PX4])]",
        "thiophosphinate_halide": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-[#6])-[F,Cl,Br,IX1]",
        ### S=PRX2
        "thiophosphonate_di_ester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-O-[!#1])-O-[!#1]",
        "dithiophosphonate_mono_ester_mono_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-O-[!#1])-[SX2]-[#6]",
        "trithiophosphonate_di_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-[SX2]-[#6])-[SX2]-[#6]",
        "thiophosphonamidate": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-O-[!#1])-[#7X3,$([NX2]=[CX3,PX4])]",
        "thiophosphondiamidate": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-[#7X3,$([NX2]=[CX3,PX4])])-[#7X3,$([NX2]=[CX3,PX4])]",
        "dithiophosphonamidate": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-[SX2]-[#6])-[#7X3,$([NX2]=[CX3,PX4])]",
        "halothiophosphonate": "[$([PX4](=[SX1])[O-,OX2]),$([PX4](=O)[SX1-,SX2]),$([PX4+](-[SX1-])[O-,OX2]),$([PX4+](-[O-])[SX1-,SX2])](-[#6])-[F,Cl,Br,IX1]",
        "halothiophosphonamidate": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-[#7X3,$([NX2]=[CX3,PX4])])-[F,Cl,Br,IX1]",
        "halodithiophosphonate": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-[S-,SX2])-[F,Cl,Br,IX1]",
        "dihalothiophosphonate": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#6])(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
        ### S=PX3
        "thiophosphate_tri_ester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-O-[!#1])(-O-[!#1])-O-[!#1]",
        "dithiophosphate_di_ester_mono_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-O-[!#1])(-O-[!#1])-[SX2]-[#6]",
        "trithiophosphate_mono_ester_di_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-O-[!#1])(-[SX2]-[#6])-[SX2]-[#6]",
        "tetrathiophosphate_tri_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[SX2]-[#6])(-[SX2]-[#6])-[SX2]-[#6]",
        "thiophosphoramidate_mono_ester": "[$([PX4](=O)(-[SX2H,SX1-])),$([PX4](-[OH,O-,OH2+])(=S)),$([PX4+](-[OH,O-,OH2+])(-[SX2H,SX1-]));!$([PX4+](-[SX2H])(-[OH,O-,OH2+]))](-O-[!#1])-[#7X3,$([NX2]=[CX3,PX4])]",
        "thiophosphoramidate_di_ester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-O-[!#1])(-O-[!#1])-[#7X3,$([NX2]=[CX3,PX4])]",
        "dithiophosphoramidate_mono_ester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[SX2H,SX2-])(-O-[!#1])-[#7X3,$([NX2]=[CX3,PX4])]",
        "dithiophosphoramidate_mono_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[OH,O-,OH2+])(-[SX2]-[#6])-[#7X3,$([NX2]=[CX3,PX4])]",
        "dithiophosphoramidate_mono_ester_mono_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-O-[!#1])(-[SX2]-[#6])-[#7X3,$([NX2]=[CX3,PX4])]",
        "trithiophosphoramidate_mono_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[SX2H,SX2-])(-[SX2]-[#6])-[#7X3,$([NX2]=[CX3,PX4])]",
        "trithiophosphoramidate_di_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[SX2]-[#6])(-[SX2]-[#6])-[#7X3,$([NX2]=[CX3,PX4])]",
        "thiophosphorodiamidate_ester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-O-[!#1])(-[#7X3,$([NX2]=[CX3,PX4])])-[#7X3,$([NX2]=[CX3,PX4])]",
        "dithiophosphorodiamidate_thioester": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[SX2]-[#6])(-[#7X3,$([NX2]=[CX3,PX4])])-[#7X3,$([NX2]=[CX3,PX4])]",
        "thiophosphoramide": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[#7X3,$([NX2]=[CX3,PX4])])(-[#7X3,$([NX2]=[CX3,PX4])])-[#7X3,$([NX2]=[CX3,PX4])]",
        "Lawesson's_reagent": "[#6]-[$([PX4]=[SX1]),$([PX4+]-[SX1-])]1-[SX2]-[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[SX2]-1)-[#6]",
        "halothiophosphate": "[$([PX4](=[SX1])[O-,OX2]),$([PX4](=O)[SX1-,SX2]),$([PX4+](-[SX1-])[O-,OX2]),$([PX4+](-[O-])[SX1-,SX2])](-[O-,OX2])-[F,Cl,Br,IX1]",
        "halodithiophosphate": "[$([PX4](=[SX1])[O-,OX2]),$([PX4](=O)[SX1-,SX2]),$([PX4+](-[SX1-])[O-,OX2]),$([PX4+](-[O-])[SX1-,SX2])](-[S-,SX2])-[F,Cl,Br,IX1]",
        "dihalothiophosphate": "[$([PX4](=[SX1])[O-,OX2]),$([PX4](=O)[SX1-,SX2]),$([PX4+](-[SX1-])[O-,OX2]),$([PX4+](-[O-])[SX1-,SX2])](-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
        "dihalodithiophosphate": "[$([PX4]=[SX1]),$([PX4+]-[SX1-])](-[SX1-,SX2])(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
        
        ## RN=P(VI)
        ### RN=PR3
        "iminophosphorane": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[#6,#1])(-[#6,#1])-[#6,#1]",
        "N-acyl_iminophosphorane": "O=C-[$([NX2]=[PX4]),$([NX2-]-[PX4+])]",
        "N-amino_iminophosphorane": "[#7]-[$([NX2]=[PX4]),$([NX2-]-[PX4+])]",
        "N-oxy_iminophosphorane": "O-[$([NX2]=[PX4]),$([NX2-]-[PX4+])]",
        ### RN=PR2X
        "iminophosphinate": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[#6,#1])(-[#6,#1])-O-[!#1]",
        "imino-thiophosphinate": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[#6,#1])(-[#6,#1])-[SX2]-[!#1]",
        "iminophosphinamide": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[#6,#1])(-[#6,#1])-[#7X3,$([NX2]=[CX3,PX4])]",
        "iminophosphorane_monohalide": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[#6,#1])(-[#6,#1])-[F,Cl,Br,IX1]",
        ### RN=PRX2
        "iminophosphonate": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[#6,#1])(-O-[!#1])-O-[!#1]",
        "iminophosphonamidate": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[#6,#1])(-O-[!#1])-[#7X3,$([NX2]=[CX3,PX4])]",
        "iminophosphondiamidate": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[#6,#1])(-[#7X3,$([NX2]=[CX3,PX4])])-[#7X3,$([NX2]=[CX3,PX4])]",
        "imino-thiophosphonate_mono_ester_mono_thioester": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[#6,#1])(-O-[!#1])-[SX2]-[!#1]",
        "imino-dithiophosphonate_di_thioester": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[#6,#1])(-[SX2]-[!#1])-[SX2]-[!#1]",
        "imino-thiophosphonamidate_thioester": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[#6,#1])(-[SX2]-[!#1])-[#7X3,$([NX2]=[CX3,PX4])]",
        "iminophosphono_halidate": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[#6,#1])(-[O-,OX2])-[F,Cl,Br,IX1]",
        "iminophosphorane_dihalide": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[#6,#1])(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
        ### RN=PX3
        "iminophosphate_mono_ester": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[O-,OH,OH2+])(-[O-,OH,OH2+])-O-[!#1]",
        "iminophosphate_di_ester": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[O-,OH,OH2+])(-O-[!#1])-O-[!#1]",
        "iminophosphate_tri_ester": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-O-[!#1])(-O-[!#1])-O-[!#1]",
        "iminophosphoramidate_mono_ester": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[OH,O-,OH2+])(-O-[!#1])-[#7X3,$([NX2]=[CX3,PX4])]",
        "iminophosphoramidate_di_ester": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-O-[!#1])(-O-[!#1])-[#7X3,$([NX2]=[CX3,PX4])]",
        "imino-thiophosphate_di_ester_mono_thioester": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-O-[!#1])(-O-[!#1])-[SX2]-[!#1]",
        "imino-dithiophosphate_mono_ester_di_thioester": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-O-[!#1])(-[SX2]-[!#1])-[SX2]-[!#1]",
        "imino-trithiophosphate_tri_thioester": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[SX2]-[!#1])(-[SX2]-[!#1])-[SX2]-[!#1]",
        "iminophosphoramidate_di_ester": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-O-[!#1])(-O-[!#1])-[#7X3,$([NX2]=[CX3,PX4])]",
        "imino-thiophosphoramidate_mono_ester_mono_thioester": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-O-[!#1])(-[SX2]-[!#1])-[#7X3,$([NX2]=[CX3,PX4])]",
        "imino-dithiophosphoramidate_di_thioester": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[SX2]-[!#1])(-[SX2]-[!#1])-[#7X3,$([NX2]=[CX3,PX4])]",
        "iminophosphorodiamidate": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-O-[!#1])(-[#7X3,$([NX2]=[CX3,PX4])])-[#7X3,$([NX2]=[CX3,PX4])]",
        "iminophosphoramide": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[#7X3,$([NX2]=[CX3,PX4])])(-[#7X3,$([NX2]=[CX3,PX4])])-[#7X3,$([NX2]=[CX3,PX4])]",
        "halo_iminophosphate": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[O-,OX2])(-[O-,OX2])-[F,Cl,Br,IX1]",
        "dihalo_iminophosphate": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[O-,OX2])(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
        "iminophosphorane_trihalide": "[$([PX4]=[NX2]),$([PX4+]-[NX2-])](-[F,Cl,Br,IX1])(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
        "phosphazene": "[PX4]=[NX2;!$(N1=[PX4]-N=[PX4]-N=[PX4]-1)]-P",
        "cyclotriphosphazene": "N1=[PX4]-N=[PX4]-N=[PX4]-1",
        ## PR5
        "λ5-phosphane": "[PX5]",
    },
    # Sulfur
    "Sulfur": {
    ## S-C, S=C
        "aliphatic_thiol": "[CX4]-[SX2H,SX1-]",
        "aromatic_thiol": "[cX3]-[SX2H,SX1-]",
        "thioenol": "[#6]=C-[SX2H,SX1-,SX3H2+]",
        "thioether": "[#6;!$(C=[C,O,S,N]);!$(C#N)]-[SX2;!$([SX2]1-[#6;X3]=,:[#6;X3]-[SX2]-C-1=C2-[SX2]-[#6;X3]=,:[#6;X3]-[SX2]-2);!r3;!$(S1-[#6;X3]~[#6;X3]~[O,S,N]~[#6;X3]~[#6;X3]-1)]-[#6;!$(C=[C,O,S,N]);!$(C#N)]", # excludes episulfides, thioenol_ethers, thioesters, thioxines, thiocyanates, tetrathiafulvalene
        "thioenol_ether": "[SX2;$(S-C=C);!$([SX2]1-[#6;X3]=,:[#6;X3]-[SX2]-C-1=C2-[SX2]-[#6;X3]=,:[#6;X3]-[SX2]-2);!$(S1-[#6;X3]~[#6;X3]~[O,S,N]~[#6;X3]~[#6;X3]-1)]",
        "O,S-acetal": "[SX2]-[CX4;!$(C(S)(O)[O,S,N,n])]-O", 
        "thioacetal": "[SX2]-[CX4;!$(C(S)(S)[O,S,N,n])]-[SX2]",
        "thioaminal": "[SX2]-[CX4;!$(C(S)(N)[O,S,N,n])]-[#7]", 
        "ketene_O,S-acetal": "C=C(-[SX2H0])-[OX2H0]",
        "ketene_thioacetal": "[C;!$(C1(-[SX2]-[#6;X3]=,:[#6;X3]-[SX2]-1)=C2-[SX2]-[#6;X3]=,:[#6;X3]-[SX2]-2)]=C(-[SX2H0])-[SX2H0]", # excludes tetrathiafulvalene
        "tetrathiafulvalene": "C1(-[SX2]-[#6;X3]=,:[#6;X3]-[SX2]-1)=C2-[SX2]-[#6;X3]=,:[#6;X3]-[SX2]-2",
        "ketene_thioaminal": "C=C(-[SX2H0])-N",
        "thioaldehyde": "[#6]-[CH](=[SX1])",
        "thioketone": "[#6]-C(=[SX1])-[#6]",
        "thioketene": "C=C=[SX1]",
        "thionium": "[CX3]=[SX2+]-[!$([#6;-])]",
        "carboxylate_thioester":  "[#6,#1]-C(=O)-[SX2;!$([SX2]1[#6](=O)[#6]1);!$([SX2]1[#6](=O)[#6]~[#6]1);!$([SX2]1[#6](=O)[#6]~[#6]~[#6]1);!$([SX2]1[#6](=O)[#6]~[#6]~[#6]~[#6]1)]-[#6;!$(C=[O,S])]",
        "thionoester": "[#6,#1]-C(=[SX1])-[O;!$(O1[#6](=S)[#6]1);!$(O1[#6](=S)[#6]~[#6]1);!$(O1[#6](=S)[#6]~[#6]~[#6]1);!$(O1[#6](=S)[#6]~[#6]~[#6]~[#6]1)]-[!$(C=[O,S])]",
        "dithioester": "[#6,#1]-C(=[SX1])-[SX2;!$([SX2]1[#6](=S)[#6]1);!$([SX2]1[#6](=S)[#6][#6]1);!$([SX2]1[#6](=S)[#6][#6][#6]1);!$([SX2]1[#6](=S)[#6][#6][#6][#6]1)]-[!$(C=[O,S])]",
        "thioamide": "[#6,#1;!$(C(=S)[#7])]-C(=[SX1])-[#7X3,$([NX2]=[CX3,PX4]);!$(N1[#6](=S)[#6]1);!$(N1[#6](=S)[#6]~[#6]1);!$(N1[#6](=S)[#6]~[#6]~[#6]1);!$(N1[#6](=S)[#6]~[#6]~[#6]~[#6]1);!$(N-[#7]);!$(N-[OH]);!$(N-O-C=O)]", # excludes dithioxamide
        "dithioxamide": "[#7]-C(=S)-C(=S)-[#7]",
        "thiohydrazide": "[!#7]-C(=[SX1])-[NX4+,#7X3,$([NX2]=[CX3,PX4])]-[#7X3,$([NX2]=[CX3,PX4])]",
        "thiohydroxamic_acid": "[#6,#1]-C(=[SX1])-[NX3]-[OH]",
        "thiohydroxamate": "[#6,#1]-C(=[SX1])-[NX3]-O-[!$(C=O)]",
        "O-acyl_thiohydroxamate": "[#6,#1]-C(=[SX1])-[NX3]-O-C(=O)",
        "thiourea[": "[#7X3,$([NX2]=[CX3,PX4])]-[C;!$(C1(=[SX1])NCCN1)](=[SX1])-[#7X3,$([NX2]=[CX3,PX4])]", # excludes thioureido ring
        "thioureido_ring": "[SX1]=C1NCCN1",
        "isothiourea": "[NX2,NX3H+]=C(-[SX2])-[#7X3,$([NX2]=[CX3,PX4])]",
        "thiocarbamate": "[#7;X3,X2]-C(=O)-[SX2]",
        "thionocarbamate": "[#7;X3,X2]-C(=[SX1])-O",
        "dithiocarbamate": "[#7;X3,X2]-C(=[SX1])-[SX2]",
        "thiocarbonate": "O-C(=O)-[SX2]",
        "thionocarbonate": "O-C(=[SX1])-O",
        "dithiocarbonate": "[SX2]-C(=O)-[SX2]",
        "xanthate": "O-C(=[SX1])-[SX2]",
        "trithiocarbonate": "[SX2]-C(=[SX1])-[SX2]",
        "imidothioate": "[#6,#1]-C(=[NX2,NX3H+;!$([NX2+]=[NX1-])])-[SX2]",
        "thioimidocarbonate": "[SX2]-C(=[NX2,NX3H+;!$([NX2+]=[NX1-])])-O",
        "dithioimidocarbonate": "[SX2]-C(=[NX2,NX3H+;!$([NX2+]=[NX1-])])-[SX2]",
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
        "disulfide": "[!#16]~[#16;X2]-,:[#16;X2]~[!#16]",
        "trisulfide": "[#16;X2]-,:[#16;X2]-,:[#16;X2]",
        "sulfenic_acid": "[#6]-[SX2]-[OH,O-,OH2+]",
        "sulfenate": "[#6]-[SX2]-O-[!#1]",
        "sulfenyl_halide": "[#6]-[SX2]-[F,Cl,Br,IX1]",
        "sulfoxylate": "O-[SX2]-O",
        "sulfenamide": "[#6]-[SX2]-[NX3]",
        "thioxime": "[#6,#1]-C(=[NX2,NX3H+]-[SX2])-[#6,#1]",
        "thiocyanate": "[*]-[SX2]-C#[NX1]",
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
        "sulfinate": "[#6]-[$([SX3]=O),$([SX3+]-[O-])]-O-[!#1]",
        "sulfinyl_halide": "[#6]-[$([SX3]=O),$([SX3+]-[O-])]-[F,Cl,Br,IX1]",
        "sulfinamide": "[#6]-[$([SX3]=O),$([SX3+]-[O-])]-[#7X3]",
        "N-sulfinyl_imine": "[#6]-[$([SX3]=O),$([SX3+]-[O-])]-[NX2]=C",
        "sulfite": "O-[$([SX3]=O),$([SX3+]-[O-])]-O",
        "amidosulfite": "O-[$([SX3]=O),$([SX3+]-[O-])]-[#7X3,$([NX2]=[CX3,PX4])]",
        "imidosulfite": "O-[$([SX3]=[NX2]),$([SX3+]-[NX2-])]-O",
        "halo_imidosulfite": "[F,Cl,Br,IX1]-[$([SX3]=[NX2]),$([SX3+]-[NX2-])]-O",
        "amidoimidosulfite": "O-[$([SX3]=[NX2]),$([SX3+]-[NX2-])]-[#7X3,$([NX2]=[CX3,PX4])]",
        "amidoimidosulfurous_halide": "[F,Cl,Br,IX1]-[$([SX3]=[NX2]),$([SX3+]-[NX2-])]-[#7X3,$([NX2]=[CX3,PX4])]",
        "sulfurous_diamide": "[#7X3,$([NX2]=[CX3,PX4])]-[$([SX3]=O),$([SX3+]-[O-])]-[#7X3,$([NX2]=[CX3,PX4])]",
        "imidosulfurous_diamide": "[#7X3,$([NX2]=[CX3,PX4])]-[$([SX3]=[NX2]),$([SX3+]-[NX2-])]-[#7X3,$([NX2]=[CX3,PX4])]",
        "halosulfite": "[F,Cl,Br,IX1]-[$([SX3]=O),$([SX3+]-[O-])]-O",
        "sulfinamide": "[#6]-[$([SX3]=O),$([SX3+]-[O-])]-[#7X3,$([NX2]=[CX3,PX4])]",
        "thiosulfite": "O-[$([SX3]=O),$([SX3+]-[O-])]-[SX2]",
        "thionosulfite": "O-[$([SX3]=[SX1]),$([SX3+]-[SX1-])]-O",
        "thiosulfinate": "[#6]-[$([SX3]=O),$([SX3+]-[O-])]-[SX2]",
        "sulfilimine": "[#6]-[$([SX3]=[NX2]),$([SX3+]-[NX2-])]-[#6]",
        "sulfinimidate": "[#6]-[$([SX3]=[NX2]),$([SX3+]-[NX2-])]-O",
        "sulfinamidine": "[#6]-[$([SX3]=[NX2]),$([SX3+]-[NX2-])]-[#7X3,$([NX2]=[CX3,PX4])]",
        "sulfinimidoyl_halide": "[#6]-[$([SX3]=[NX2]),$([SX3+]-[NX2-])]-[F,Cl,Br,IX1]",
        "imidothionyl_halide": "[F,Cl,Br,IX1]-[$([SX3]=[NX2]),$([SX3+]-[NX2-])]-[F,Cl,Br,IX1]",
        ### tetravalent
        "λ4-sulfane": "[SX4](-[*])(-[*])(-[*])-[*]",
        "sulfone": "[#6]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[#6]",
        "sulfone": "[#6]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[#6]",
        "sulfonic_acid": "[#6]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[OH,O-,OH2+]",
        "sulfonate": "[#6]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-O-[!#1;!#7]",
        "sulfonyl_halide": "[#6]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[F,Cl,Br,IX1]",
        "thiosulfonate": "[#6]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[SX2]",
        "sulfonamide": "[#6]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[#7X3;!$(NC(=O)N);!$([#7]([$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])])-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])])]", # excludes sulfonimide, sulfonylurea
        "N-sulfonyl_imine": "[#6]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[NX2]=[CX3]",
        "N-sulfonyl_iminophosphorane": "[#6]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[NX2]=[PX4]",
        "N-sulfonate": "[#6]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-O-[#7]",
        "sulfonimide": "[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[#7;X3]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[#6]",
        "sulfamide": "[#7X3,$([NX2]=[CX3,PX4])]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[#7X3,$([NX2]=[CX3,PX4])]",
        "sulfamic_acid": "[#7X3,$([NX2]=[CX3,PX4])]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[OH,O-,OH2+]",
        "sulfamate": "[#7X3,$([NX2]=[CX3,PX4])]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-O",
        "sulfamoyl_halide": "[#7X3,$([NX2]=[CX3,PX4])]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[F,Cl,Br,IX1]",
        "sulfate_mono_substituted": "[OH,O-,OH2+]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-O-[!#1]",
        "sulfate_di_substituted": "[!#1]-O-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-O-[!#1]",
        "fluorosulfate": "F-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-O",
        "halosulfate": "[Cl,Br,IX1]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-O",
        "Bunte_salt": "[OH,O-,OH2+]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[SX2]-[!#1]",
        "sulfoximine": "[#6]-[$([SX4](=O)=[NX2]),$([SX4+](=O)-[NX2-]),$([SX4+](-[O-])=[NX2]),$([SX4+2](-[O-])-[NX2-])]-[#6]",
        "sulfondiimine": "[#6]-[$([SX4](=[NX2])=[NX2]),$([SX4+](=[NX2])-[NX2-]),$([SX4+2](-[NX2-])-[NX2-])]-[#6]",
        "sulfonimidic_acid": "[#6]-[$([SX4](=O)=[NX2]),$([SX4+](=O)-[NX2-]),$([SX4+](-[O-])=[NX2]),$([SX4+2](-[O-])-[NX2-])]-[O-,OH,OH2+]",
        "sulfonimidate": "[#6]-[$([SX4](=O)=[NX2]),$([SX4+](=O)-[NX2-]),$([SX4+](-[O-])=[NX2]),$([SX4+2](-[O-])-[NX2-])]-O",
        "sulfondiimidate": "[#6]-[$([SX4](=[NX2])=[NX2]),$([SX4+](=[NX2])-[NX2-]),$([SX4+2](-[NX2-])-[NX2-])]-O",
        "sulfonimidoyl_halide": "[#6]-[$([SX4](=O)=[NX2]),$([SX4+](=O)-[NX2-]),$([SX4+](-[O-])=[NX2]),$([SX4+2](-[O-])-[NX2-])]-[F,Cl,Br,IX1]",
        "sulfondiimidoyl_halide": "[#6]-[$([SX4](=[NX2])=[NX2]),$([SX4+](=[NX2])-[NX2-]),$([SX4+2](-[NX2-])-[NX2-])]-[F,Cl,Br,IX1]",
        "sulfonimidamide": "[#6]-[$([SX4](=O)=[NX2]),$([SX4+](=O)-[NX2-]),$([SX4+](-[O-])=[NX2]),$([SX4+2](-[O-])-[NX2-])]-[#7X3,$([NX2]=[CX3,PX4])]",
        "sulfondiimidamide": "[#6]-[$([SX4](=[NX2])=[NX2]),$([SX4+](=[NX2])-[NX2-]),$([SX4+](-[NX2-])=[NX2])]-[#7X3,$([NX2]=[CX3,PX4])]",
        "sulfamimidoyl_halide": "[F,Cl,Br,IX1]-[$([SX4](=O)=[NX2]),$([SX4+](=O)-[NX2-]),$([SX4+](-[O-])=[NX2]),$([SX4+2](-[O-])-[NX2-])]-[#7X3,$([NX2]=[CX3,PX4])]",
        "sulfamdiimidoyl_halide": "[F,Cl,Br,IX1]-[$([SX4](=[NX2])=[NX2]),$([SX4+](=[NX2])-[NX2-]),$([SX4+2](-[NX2-])-[NX2-])]-[#7X3,$([NX2]=[CX3,PX4])]",
        "imidosulfate": "O-[$([SX4](=O)=[NX2]),$([SX4+](=O)-[NX2-]),$([SX4+](-[O-])=[NX2]),$([SX4+2](-[O-])-[NX2-])]-O",
        "halo_imidosulfate": "[F,Cl,Br,IX1]-[$([SX4](=O)=[NX2]),$([SX4+](=O)-[NX2-]),$([SX4+](-[O-])=[NX2]),$([SX4+2](-[O-])-[NX2-])]-O",
        "imidosulfuric_monoamide": "O-[$([SX4](=O)=[NX2]),$([SX4+](=O)-[NX2-]),$([SX4+2](-[O-])-[NX2-]),$([SX4+](-[O-])=[NX2])]-[#7X3,$([NX2]=[CX3,PX4])]",
        "imidosulfuric_diamide": "[#7X3,$([NX2]=[CX3,PX4])]-[$([SX4](=O)=[NX2]),$([SX4+](=O)-[NX2-]),$([SX4+2](-[O-])-[NX2-]),$([SX4+](-[O-])=[NX2])]-[#7X3,$([NX2]=[CX3,PX4])]",
        "diimidosulfate": "O-[$([SX4](=[NX2])=[NX2]),$([SX4+](=[NX2])-[NX2-]),$([SX4+2](-[NX2-])-[NX2-])]-O",
        "diimidosulfuric_monoamide": "O-[$([SX4](=[NX2])=[NX2]),$([SX4+](=[NX2])-[NX2-]),$([SX4+2](-[NX2-])-[NX2-])]-[#7X3,$([NX2]=[CX3,PX4])]",
        "diimidosulfuric_diamide": "[#7X3,$([NX2]=[CX3,PX4])]-[$([SX4](=[NX2])=[NX2]),$([SX4+](=[NX2])-[NX2-]),$([SX4+2](-[NX2-])-[NX2-])]-[#7X3,$([NX2]=[CX3,PX4])]",
        "halo_diimidosulfate": "[F,Cl,Br,IX1]-[$([SX4](=[NX2])=[NX2]),$([SX4+](=[NX2])-[NX2-]),$([SX4+2](-[NX2-])-[NX2-])]-O",
        "imidosulfuryl_halide": "[F,Cl,Br,IX1]-[$([SX4](=O)=[NX2]),$([SX4+](=O)-[NX2-]),$([SX4+2](-[O-])-[NX2-]),$([SX4+](-[O-])=[NX2])]-[F,Cl,Br,IX1]",
        "dihalo_sulfur_diimide": "[F,Cl,Br,IX1]-[$([SX4](=[NX2])=[NX2]),$([SX4+](=[NX2])-[NX2-]),$([SX4+2](-[NX2-])-[NX2-])]-[F,Cl,Br,IX1]",
        "trifluorosulfanyl": "[*]-[SX4](-F)(-F)-F",
        "sulfoxonium": "[#6;!-]-[$([SX4+]=O),$([SX4+2]-[O-])](-[#6;!-])-[#6;!-]", # excludes sulfoxonium ylides
        ### common sulfonyls
        "mesyl": "[*]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[CH3]",
        "o-tosyl": "[*]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[cH0]1:[cH]:[cH]:[cH]:[cH]:[cH0](-[CH3]):1",
        "p-tosyl": "[*]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[cH0]1:[cH]:[cH]:[cH0](-[CH3]):[cH]:[cH]:1",
        "o-nosyl": "[*]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[cH0]1:[cH0](-[N+](=O)(-[O-])):[cH]:[cH1]:[cH]:[cH]:1",
        "p-nosyl": "[*]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[cH0]1:[cH]:[cH]:[cH0](-[N+](=O)(-[O-])):[cH]:[cH]:1",
        "bresyl": "[*]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[cH0]1:[cH]:[cH]:[cH0](-Br):[cH]:[cH]:1",
        "triflyl": "[*]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-C(F)(F)F",
        "dansyl": "[*]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[cH0]1:[cH]:[cH]:[cH]:[cH0]2:[cH0](-[NX3](-[CH3])-[CH3]):[cH]:[cH]:[cH]:[cH0]:1:2", # 5-dimethylaminonaphthalene
        "nonaflyl": "[*]-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)F", # C4F9
        ### hexavalent
        "λ6-sulfane": "[SX6]",
        "pentafluorosulfanyl": "[*]-S(F)(F)(F)(F)F",
    },
    # Halogens
    "Halogen": {
        ## C(sp3)-X
        "alkyl_fluoride": "F-[CX4;!$(C(F)(C(F)(F)F)C(F)(F)F)](-[!$([F,Cl,Br,I])])(-[!$([F,Cl,Br,I])])-[!$([F,Cl,Br,I])]", # excludes perfluoro-isopropyl
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
        "imidoyl_fluoride": "[!$([O,N,S])]-C(=[NX2,NX3H+;!r3])-F",
        "imidoyl_chloride": "[!$([O,N,S])]-C(=[NX2,NX3H+;!r3])-Cl",
        "imidoyl_bromide": "[!$([O,N,S])]-C(=[NX2,NX3H+;!r3])-Br",
        "imidoyl_iodide": "[!$([O,N,S])]-C(=[NX2,NX3H+;!r3])-[IX1]",
        "halo_formate": "[F,Cl,Br,IX1]-C(=O)-O",
        "halo_formamide": "[F,Cl,Br,IX1]-C(=O)-[#7]",
        "halo_formamidine": "[F,Cl,Br,IX1]-C(=[NX2,NX3+])-[NX3]",
        "halo_formimidate": "[F,Cl,Br,IX1]-C(=[NX2,NX3+;!$([NX2+]=[NX1-])])-O",
        "halo_formamidoxime": "[F,Cl,Br,IX1]-C(=[NX2,NX3+]-O)-[NX3]",
        "halo_formamidrazone": "[F,Cl,Br,IX1]-[CX3](-,=N-[NX3])-,=N",
        "halo_thioformate":  "[F,Cl,Br,IX1]-C(=O)-[SX2]",
        "halo_thionoformate": "[F,Cl,Br,IX1]-C(=[SX1])-O",
        "halo_dithioformate": "[F,Cl,Br,IX1]-C(=[SX1])-[SX2]",
        "halo_thioformamide": "[F,Cl,Br,IX1]-C(=[SX1])-[#7]",
        "halo_thioformimidate": "[F,Cl,Br,IX1]-C(=[NX2,NX3+;!$([NX2+]=[NX1-])])-[SX2]",
        "phosgene_oxime": "[F,Cl,Br,IX1]-C(=[NX2,NX3+]-O)-[F,Cl,Br,IX1]",
        "phosgene_hydrazone": "[F,Cl,Br,IX1]-C(=[NX2,NX3+]-[#7])-[F,Cl,Br,IX1]",
        ## C(sp)-X
        "alkynyl_fluoride": "C#[CX2]-F",
        "alkynyl_chloride": "C#[CX2]-Cl",
        "alkynyl_bromide": "C#[CX2]-Br",
        "alkynyl_iodide": "C#[CX2]-[IX1]",
        ## 2+ X
        "dihalo_methylidene": "[F,Cl,Br,IX1]-[CX3](-[F,Cl,Br,IX1])=[*]",
        "geminal_dihalide": "[!$([F,Cl,Br,I])]-[CX4;!$(C(F)(F)-C(F)(F));!$([CH](F)F)](-[F,Cl,Br,IX1])(-[F,Cl,Br,IX1])-[!$([F,Cl,Br,I])]",
        "vicinal_dihalide": "[F,Cl,Br,IX1]-[CX4](-[!$([F,Cl,Br,I])])(-[!$([F,Cl,Br,I])])-[CX4](-[!$([F,Cl,Br,I])])(-[!$([F,Cl,Br,I])])-[F,Cl,Br,IX1]",
        "difluoromethyl": "[*]-[CH]([#1])(F)F",
        "trifluoromethyl": "[!$(C([F,#1,$(C(F)(F)F)])(C(F)(F)F)C(F)(F)F);!$([cH0]1[cH][cH0](-C(F)(F)F)[cH][cH0][cH]1);!$([cH0]1[cH0]c(-C(F)(F)F)[cH]c(-C(F)(F)F)[cH]1);!$([cH0]1[cH]c(-C(F)(F)F)[cH0]c(-C(F)(F)F)[cH]1);!$(C(F)(F));!$(C(=O)O);!$([$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])])]-C(F)(F)F", # excludes perfluoro-isopropyl, hexafluoro-isopropyl, perfluoro-t-butyl, triflate, CF3-phenyls
        "trifluoroacetoxy": "FC(F)(F)-C(=O)-O-[!#1]",
        "trihalomethyl": "[*]-[C;!$(C(F)(F)F)](-[F,Cl,Br,IX1])(-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]", # excluding trifluoromethyl
        "2,4,6-trifluorophenyl": "[*]-[cH0]1:[cH0](-F):[cH]:[cH0](-F):[cH]:[cH0](-F):1",
        "pentafluorophenyl": "[*]-c1c(F)c(F)c(F)c(F)c1(F)",
        "3,5-bis(trifluoromethyl)phenyl": "[*]-[cH0]1:[cH]:[cH0](-C(F)(F)F):[cH]:[cH0](-C(F)(F)F):[cH]:1",
        "2,4,6-tris(trifluoromethyl)phenyl": "[*]-[cH0]1:[cH0](-C(F)(F)F):[cH]:[cH0](-C(F)(F)F):[cH]:[cH0](-C(F)(F)F):1",
        ### PFAs
        "perfluoro-ethyl": "[!$(C(F)(F))]-C(F)(F)-C(F)(F)F",
        "perfluoro-ethylene": "[!F;!$(C(F)(F))]-C(F)(F)-C(F)(F)-[!F;!$(C(F)(F))]",
        "perfluoro-propyl": "[!$(C(F)(F))]-C(F)(F)-C(F)(F)-C(F)(F)F",
        "perfluoro-propylene": "[!F;!$(C(F)(F))]-C(F)(F)-C(F)(F)-C(F)(F)-[!F;!$(C(F)(F))]",
        "hexafluoro-isopropyl": "[CH](-C(F)(F)F)-C(F)(F)F",
        "perfluoro-isopropyl": "[!$(C(F)(F))]-C(F)(-C(F)(F)F)-C(F)(F)F",
        "perfluoro-butyl": "[!$(C(F)(F))]-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)F",
        "perfluoro-t-butyl": "[*]-C(C(F)(F)F)(C(F)(F)F)C(F)(F)F",
        "perfluoro-butylene": "[!F;!$(C(F)(F))]-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-[!F;!$(C(F)(F))]",
        "perfluoro-pentyl": "[!$(C(F)(F))]-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)F",
        "perfluoro-pentylene": "[!F;!$(C(F)(F))]-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-[!F;!$(C(F)(F))]",
        "perfluoro-hexyl": "[!$(C(F)(F))]-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)F",
        "perfluoro-hexylene": "[!F;!$(C(F)(F))]-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-[!F;!$(C(F)(F))]",
        "perfluoro-cyclohexyl": "[*]-C1-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)1",
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
    },
    # Arsenic
    "Arsenic": {
        "arsine": "[#6,#14,#1]-[AsX3;!$([As]1-[#6;X3]-,:[#6;X3]-[#6;X3]-,:[#6;X3]-1)](-[#6,#14,#1])-[#6,#14,#1]", # excludes arsole
        "arsole": "[AsX3]1-[#6;X3]-,:[#6;X3]-[#6;X3]-,:[#6;X3]-1",
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
        "aminoarsine": "[#6,#14,#1]-[AsX3](-[#6,#14,#1])-[#7X3,$([NX2]=[CX3,PX4])]",
        "diaminoarsine": "[#6,#14,#1]-[AsX3](-[#7X3,$([NX2]=[CX3,PX4])])-[#7X3,$([NX2]=[CX3,PX4])]",
        "triaminoarsine": "[#7X3,$([NX2]=[CX3,PX4])]-[AsX3](-[#7X3,$([NX2]=[CX3,PX4])])-[#7X3,$([NX2]=[CX3,PX4])]",
        "arsinic_acid": "[#6]-[$([AsX4]=O),$([AsX4+]-[O-])](-[#6])-[O-,OH,OH2+]",
        "arsinyl_halide": "[#6]-[$([AsX4]=O),$([AsX4+]-[O-])](-[#6])-[F,Cl,Br,IX1]",
        "arsinate": "[#6]-[$([AsX4]=O),$([AsX4+]-[O-])](-[#6])-O-[!#1]",
        "arsonic_acid": "[#6]-[$([AsX4]=O),$([AsX4+]-[O-])](-[O-,OH,OH2+])-[O-,OH,OH2+]",
        "arsenyl_dihalide": "[#6]-[$([AsX4]=O),$([AsX4+]-[O-])](-[F,Cl,Br,IX1])-[F,Cl,Br,IX1]",
        "arsonate": "[#6]-[$([AsX4]=O),$([AsX4+]-[O-])](-O)-O",
        "arsenite": "O-[AsX3](-O)-O",
        "monothio_arsenite": "O-[AsX3](-O)-[SX2,SX1-]",
        "dithio_arsenite": "O-[AsX3](-[SX2,SX1-])-[SX2,SX1-]",
        "trithio_arsenite": "[SX2,SX1-]-[AsX3](-[SX2,SX1-])-[SX2,SX1-]",
        "arsenate": "O-[$([AsX4]=O),$([AsX4+]-[O-])](-O)-O",
        "thiono_arsenate": "[$([AsX4]=[SX1]),$([AsX4+]-[SX1-])](-[OX2])(-[OX2])-[OX2]",
        "monothio_arsenate": "[$([AsX4]=O),$([AsX4+]-[O-])](-O)(-O)-[SX2,SX1-]",
        "dithio_arsenate": "[$([AsX4]=O),$([AsX4+]-[O-])](-O)(-[SX2,SX1-])-[SX2,SX1-]",
        "trithio_arsenate": "[$([AsX4]=O),$([AsX4+]-[O-])](-[SX2,SX1-])(-[SX2,SX1-])(-[SX2,SX1-])",
        "tetrathio_arsenate": "[$([AsX4]=[SX1]),$([AsX4+]-[SX1-])](-[SX2,SX1-])(-[SX2,SX1-])-[SX2,SX1-]",
    },
    # Selenium
    "Selenium": {
        "selenoether": "[#6;!$(C#N)]-[SeX2;!r3]-[#6;!$(C#N)]",
        "selenophene": "[#34;X2]1:c:c:c:c1",
        "diselenide": "[#34;X2]-,:[#34;X2]",
        "selenosulfide": "[#34;X2]-,:[#16;X2]",
        "selenol": "[#6;!$(C=[O,S,N])]-[SeX2H]",
        "selenamide": "[*]-[SeX2]-[#7]",
        "selenoester": "[#6,#1]-C(=O)-[SeX2]",
        "selanyl_halide": "[SeX2]-[F,Cl,Br,IX1]",
        "selenoxide": "[#6]-[$([SeX3]=O),$([SeX3+]-[O-])]-[#6]",
        "selenonium": "[!-]-[SeX3+](-[!-])-[!-]",
        "selenonium_ylide": "[$([SeX3+]-[C-]),$([SeX3]=C)](-[#6])-[#6]",
        "seleninic_acid": "[#6]-[$([SeX3]=O),$([SeX3+]-[O-])]-[O-,OH,OH2+]",
        "seleninate": "[#6]-[$([SeX3]=O),$([SeX3+]-[O-])]-O",
        "selenonic_acid": "[#6]-[$([SeX4](=O)=O),$([SeX4+](=O)-[O-]),$([SeX4+2](-[O-])-[O-])]-[O-,OH,OH2+]",
        "selenonate": "[#6]-[$([SeX4](=O)=O),$([SeX4+](=O)-[O-]),$([SeX4+2](-[O-])-[O-])]-O",
        "selenocyanate": "[*]-[SeX2]-[CX2]#[NX1]",
        "isoselenocyanate": "[*]-[NX2]=[CX2]=[SeX1]",
        "λ4-selenane": "[SeX4](-[*])(-[*])(-[*])-[*]", 
        "λ6-selenane": "[SeX4](-[*])(-[*])(-[*])(-[*])-[*]", 
    }

}

# [c;!$(c(:a)(:a)(:a))] means an aromatic C that is NOT connected to 3 aromatic atoms via aromatic bonds.
# This prevents additional fused aromatic rings.

HOMOAROMATICS: Dict[str, str] = {
    # 1 ring
    "Monocyclic": {
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
        "o-phenylene": "[!$([CH3]);!$([OH]);!$(O[CH3])]-[cH0;!$(c12ccccc1-ccc-2);!$(c1([N+](=O)[O-])ccccc1-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])])]1:[cH0;!$(c1([N+](=O)[O-])ccccc1-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])])](-[!$([CH3]);!$([OH]);!$(O[CH3])]):[cH]:[cH]:[cH]:[cH]:1", # excludes o-tolyl, o-tosyl, o-nosyl, o-hydroxyphenyl, fluoranthene
        "m-phenylene": "[!$([CH3]);!$([OH]);!$(O[CH3])]-[cH0]1:[cH]:[cH0](-[!$([CH3]);!$([OH]);!$(O[CH3])]):[cH]:[cH]:[cH]:1", # excludes m-tolyl
        "p-phenylene": "[!$([CH3]);!$([OH]);!$(O[CH3])]-[cH0;!$(c1([N+](=O)[O-])ccc(-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])])cc1)]1:[cH]:[cH]:[cH0;!$(c1([N+](=O)[O-])ccc(-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])])cc1)](-[!$([CH3]);!$([OH]);!$(O[CH3])]):[cH]:[cH]:1", # excludes p-tolyl, p-tosyl, p-nosyl
        "biphenyl": "[c;!$(c~c1ccccc1);!$(c(:a)(:a):a)]1[c;!$(c~c1ccccc1);!$(c(:a)(:a):a)][c;!$(c~c1ccccc1);!$(c(:a)(:a):a)][c;!$(c~c1ccccc1);!$(c(:a)(:a):a)][c;!$(c~c1ccccc1);!$(c(:a)(:a):a)][cX3H0;!$(c1cc~[#6]~c1);!$(c1c-cc-1)]1-[cX3H0;!$(c1cc~[#6]~c1);!$(c1c-cc-1)]1[c;!$(c~c1ccccc1);!$(c(:a)(:a):a)][c;!$(c~c1ccccc1);!$(c(:a)(:a):a)][c;!$(c~c1ccccc1);!$(c(:a)(:a):a)][c;!$(c~c1ccccc1);!$(c(:a)(:a):a)][c;!$(c~c1ccccc1);!$(c(:a)(:a):a)]1", # excludes terphenyls, biphenylene, fluorene, etc.
        "biaryl": "[cX3H0;!$(c1cc~[#6]~c1);!$(c1c-cc-1);!$([cX3H0]1[c;!$(c~c1ccccc1);!$(c(:a)(:a):a)][c;!$(c~c1ccccc1);!$(c(:a)(:a):a)][c;!$(c~c1ccccc1);!$(c(:a)(:a):a)][c;!$(c~c1ccccc1);!$(c(:a)(:a):a)][c;!$(c~c1ccccc1);!$(c(:a)(:a):a)]1)]-[cX3H0,nX3+;!$(c1cc~[#6]~c1);!$(c1c-cc-1)]",
        "o-terphenyl": "c1cccc[c;!$(c1cc~[#6]~c1);!$(c1c-cc-1)]1-[c;!$(c1cc~[#6]~c1);!$(c1c-cc-1)]1[c;!$(c1cc~[#6]~c1);!$(c1c-cc-1)](-[c;!$(c1cc~[#6]~c1);!$(c1c-cc-1)]2ccccc2)cccc1",
        "m-terphenyl": "c1cccc[c;!$(c1cc~[#6]~c1);!$(c1c-cc-1)]1-[c;!$(c1cc~[#6]~c1);!$(c1c-cc-1)]1c[c;!$(c1cc~[#6]~c1);!$(c1c-cc-1)](-[c;!$(c1cc~[#6]~c1);!$(c1c-cc-1)]2ccccc2)ccc1",
        "p-terphenyl": "c1cccc[c;!$(c1cc~[#6]~c1);!$(c1c-cc-1)]1-[c;!$(c1cc~[#6]~c1);!$(c1c-cc-1)]1cc[c;!$(c1cc~[#6]~c1);!$(c1c-cc-1)](-[c;!$(c1cc~[#6]~c1);!$(c1c-cc-1)]2ccccc2)cc1",
        "benzyne": "c1#ccccc1",
        "tropylium": "[c+]1cccccc1",
    },
    # !$(c1:c~[#6]~[#6]~c:1) prevents 5-membered rings at bridgeheads. Used to exclude acenaphthylene, fluoranthene, etc.
    # !$([cR2;r6]) prevents fusing with 6-membered rings but allows fusing with 5-membered rings.
    # !$(c12c3cccc1ccc(c24)cccc4cc3) prevents a pyrene-type fusin on two adjacent bridgeheads
    # 2 rings
    "Bicyclic": {
        "indene": "[CX4]1C=C[cX3H0]2:c:c:c:c:[cX3H0]:2-1",
        "naphthalene": "[c;!$([cR2;r6](:a)(:a)(:a));!$(c1c~c2cccc3cccc1c32)]1:[c;!$([cR2;r6](:a)(:a)(:a));!$(c1c~c2cccc3cccc1c32)]:[c;!$([cR2;r6](:a)(:a)(:a));!$(c1c~c2cccc3cccc1c32)]:[c;!$([cR2;r6](:a)(:a)(:a));!$(c1c~c2cccc3cccc1c32)]:[cX3H0;!$(c1:c~[#6]~[#6]~c:1)]2:[c;!$([cR2;r6](:a)(:a)(:a));!$(c1c~c2cccc3cccc1c32)]:[c;!$([cR2;r6](:a)(:a)(:a));!$(c1c~c2cccc3cccc1c32)]:[c;!$([cR2;r6](:a)(:a)(:a));!$(c1c~c2cccc3cccc1c32)]:[c;!$([cR2;r6](:a)(:a)(:a));!$(c1c~c2cccc3cccc1c32)]:[cX3H0;!$(c1:c~[#6]~[#6]~c:1)]:1:2", # excludes acenaphthene, acenaphthylene, benzofluoranthenes
        "azulene": "c1:c:c:c2:c:c:c:c:c:c:1-2",
    },
    # 3 rings
    "Tricyclic": {
    "anthracene": "[c;!$(c12ccccc2cccc1)]1:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0;!$(c1:c~[#6]~[#6]~c:1)]2:[c;!$(c(:a)(:a)(:a))]:[cX3H0;!$(c1:c~[#6]~[#6]~c:1)]3:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0;!$(c1:c~[#6]~[#6]~c:1)]:3:[c;!$(c(:a)(:a)(:a))]:[cX3H0;!$(c1:c~[#6]~[#6]~c:1)]:1:2",
    "phenanthrene": "[c;!$(c12ccccc2cccc1)]1:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0]2:[cX3H0]3:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0;!$(c1:c~[#6]~[#6]~c:1)]:3:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0;!$(c1:c~[#6]~[#6]~c:1)]:1:2",   
    "biphenylene": "c1:c:c:c:[cX3H0]2-[cX3H0]3:c:c:c:c:[cX3H0]:3-[cX3H0]:1:2",
    "fluorene": "c1:c:c:c:[cX3H0]2-[cX3H0]3:c:c:c:c:[cX3H0]:3-[CX4;!$([CH]-[CH2]-O-[$(C(=O)(O)[!#6])])]-[cX3H0]:1:2",
    "acenaphthylene": "[c;!$(c1ccccc1)]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0](-[#6;X3;!$(c1ccccc1)]=,:[#6;X3;!$(c1ccccc1)]3):[cX3H0]2:[cX3H0]-3:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0;!$(c1:c~[#6]~[#6]~c:1);!$(c12c3cccc1ccc(c24)cccc4cc3)]:1:2", # excludes fluoranthene, aceanthrylene, acephenanthrylene, pyracyclene, cyclopenta[cd]pyrene, indeno[cd]pyrene
    "acenaphthene": "[c;!$(c1ccccc1)]1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0](-C-C3):[cX3H0]2:[cX3H0]-3:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0;!$(c1:c~[#6]~[#6]~c:1);!$(c12c3cccc1ccc(c24)cccc4cc3)]:1:2", # excludes aceanthrene, acephenanthrene, pyracyclene
    },
    # 4 rings
    "Tetracyclic": {
    "tetracene":  "[c;!$(c12ccccc2cccc1)]1:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0]2:c:[cX3H0]3:c:[cX3H0]4:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0]:4:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "tetraphene": "[c;!$(c12ccccc2cccc1)]1:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0]2:c:[cX3H0;!$(c12c3cccc1ccc(c24)cccc4cc3)]3:[cX3H0;!$(c12c3cccc1ccc(c24)cccc4cc3)]4:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0]:4:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0]:3:c:[cX3H0]:1:2",
    "chrysene":   "[c;!$(c12ccccc2cccc1)]1:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0;!$(c12c3cccc1ccc(c24)cccc4cc3)]2:[cX3H0;!$(c12c3cccc1ccc(c24)cccc4cc3)]3:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0]4:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0;!$(c12c3cccc1ccc(c24)cccc4cc3)]:4:[cX3H0;!$(c12c3cccc1ccc(c24)cccc4cc3)]:3:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0]:1:2",
    "benzo[c]phenanthrene": "[c;!$(c12ccccc2cccc1)]1:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0]2:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0]3:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0]4:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0]:4:[cX3H0;!$(c1:c:c-C-c:c:1);!$(c1:c:c-c:c:1)]:3:[cX3H0]:1:2",
    "pyrene": "[c;!$(c12ccccc2cccc1)]1:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0;!$(c1:c~[#6]~[#6]~c:1)](:[cX3H0]2:[cX3H0]34):[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0;!$(c1:c~[#6]~[#6]~c:1)]:4:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0;!$(c1:c~[#6]~[#6]~c:1)]:3:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0;!$(c1:c~[#6]~[#6]~c:1)]:1:2",   
    "triphenylene": "[c;!$(c12ccccc2cccc1)]1:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0;!$(c12c3cccc1ccc(c24)cccc4cc3)]2:[cX3H0;!$(c12c3cccc1ccc(c24)cccc4cc3)]3:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0;!$(c12c3cccc1ccc(c24)cccc4cc3)]:3:[cX3H0;!$(c12c3cccc1ccc(c24)cccc4cc3)]4:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0;!$(c12c3cccc1ccc(c24)cccc4cc3)]:4:[cX3H0;!$(c12c3cccc1ccc(c24)cccc4cc3)]:1:2",   
    "fluoranthene": "[c;!$(c12ccccc2cccc1)]1:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0](-[cX3H0]4:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0]5:4):[cX3H0]2:[cX3H0]-5:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0]:1:2",
    "aceanthrylene": "c1:c:c:[cX3H0](-[#6;X3;!$(c1ccccc1)]=,:[#6;X3;!$(c1ccccc1)]3):[cX3H0]2:[cX3H0]-3:[cX3H0]4:c:c:c:c:[cX3H0]:4:c:[cX3H0;!$(c1:c~[#6]~[#6]~c:1)]:1:2",
    "aceanthrene": "c1:c:c:[cX3H0](-C-C3):[cX3H0]2:[cX3H0]-3:[cX3H0]4:c:c:c:c:[cX3H0]:4:c:[cX3H0;!$(c1:c~[#6]~[#6]~c:1)]:1:2",
    "acephenanthrylene": "c1:c:c:c:[cX3H0;!$(c12c3cccc1ccc(c24)cccc4cc3)]2:[cX3H0;!$(c12c3cccc1ccc(c24)cccc4cc3)]3:c:c:c:c(-[#6;X3;!$(c1ccccc1)]=,:[#6;X3;!$(c1ccccc1)]4):[cX3H0]:3:c-4:c:[cX3H0]:1:2", 
    "acephenanthrene": "c1:c:c:c:[cX3H0;!$(c12c3cccc1ccc(c24)cccc4cc3)]2:[cX3H0;!$(c12c3cccc1ccc(c24)cccc4cc3)]3:c:c:c:c(-C-C4):[cX3H0]:3:c-4:c:[cX3H0]:1:2",
    "pyracyclene":  "[cX3H0]14:c:c:[cX3H0](~[#6]~[#6]3):[cX3H0]2:[cX3H0]~3:c:c:[cX3H0](~[#6]~[#6]~4):c:1:2",

    },
    # 5 rings
    "Pentacyclic": {
    "pentacene": "c1:c:c:c:[cX3H0]2:c:[cX3H0]3:c:[cX3H0]4:c:[cX3H0]5:c:c:c:c:[cX3H0]:5:c:[cX3H0]:4:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "pentaphene": "c1:c:c:c:[cX3H0]2:c:[cX3H0]3:[cX3H0]4:c:[cX3H0]5:c:c:c:c:[cX3H0]:5:c:[cX3H0]:4:c:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "pentahelicene": "[c;!$(c12ccccc2cccc1)]1:[c;!$(c12ccccc2cccc1)]:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:[cX3H0]4:c:c:[cX3H0]5:c:c:[c;!$(c12ccccc2cccc1)]:[c;!$(c12ccccc2cccc1)]:[cX3H0]:5:[cX3H0]:4:[cX3H0]:3:[cX3H0]:1:2",
    "picene": "c1:c:c:c:[cX3H0]2:c:c:[cX3H0]3:[cX3H0]4:c:c:[cX3H0]5:c:c:c:c:[cX3H0]:5:[cX3H0]:4:c:c:[cX3H0]3:[cX3H0]:1:2",
    "perylene": "[cX3H0]12:[cX3H0]3:c:c:c:[cX3H0]:1:c:c:c:[cX3H0]:2:[cX3H0]4:c:c:c:[cX3H0]5:c:c:c:[cX3H0]:3:[cX3H0]:4:5",
    "benzo[a]tetracene": "c1:c:c:c:[cX3H0]2:c:[cX3H0]3:c:[cX3H0]4:[cX3H0]5:c:c:c:c:[cX3H0]:5:c:c:[cX3H0]:4:c:[cX3H0]:3:c:[cX3H0]:1:2",
    "benzo[a]pyrene": "c1:[cX3H0]5:c:c:c:c:[cX3H0]:5:[cX3H0](:[cX3H0]2:[cX3H0]34):c:c:[cX3H0]:4:c:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2",  
    "benzo[e]pyrene": "c1:c:c:[cX3H0](:[cX3H0]2:[cX3H0]34):[cX3H0]5:c:c:c:c:[cX3H0]:5:[cX3H0]:4:c:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2",  
    "benzo[b]chrysene": "[cX3H0]12:c:c:c:c:[cX3H0]:2:c:c:[cX3H0]3:[cX3H0]4:c:[cX3H0]5:c:c:c:c:[cX3H0]:5:c:[cX3H0]:4:c:c:[cX3H0]:3:1",
    "benzo[c]chrysene": "c1:c:c:c:[cX3H0]2:c:c:[cX3H0]3:[cX3H0]4:[cX3H0]5:c:c:c:c:[cX3H0]:5:c:c:[cX3H0]:4:c:c:[cX3H0]:3:[cX3H0]:1:2",
    "benzo[g]chrysene": "c1:c:c:c:[cX3H0]2:c:c:[cX3H0]3:[cX3H0]4:c:c:c:c:[cX3H0]:4:[cX3H0]5:c:c:c:c:[cX3H0]:5:[cX3H0]:3:[cX3H0]:1:2",
    "benzo[b]triphenylene" : "c1:c:c:c:[cX3H0]2:c:[cX3H0]3:[cX3H0]4:c:c:c:c:[cX3H0]:4:[cX3H0]5:c:c:c:c:[cX3H0]:5:[cX3H0]:3:c:[cX3H0]:1:2",
    "dibenz[a,h]anthracene": "c1:c:c:c:[cX3H0]2:c:c:[cX3H0]3:c:[cX3H0]4:[cX3H0]5:c:c:c:c:[cX3H0]:5:c:c:[cX3H0]:4:c:[cX3H0]:3:[cX3H0]:1:2",
    "dibenz[a,j]anthracene": "c1:c:c:c:[cX3H0]2:c:c:[cX3H0]3:c:[cX3H0]4:c:c:[cX3H0]5:c:c:c:c:[cX3H0]:5:[cX3H0]:4:c:[cX3H0]:3:[cX3H0]:1:2",
    "dibenzo[b,g]phenanthrene": "c1:c:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:[cX3H0]4:c:[cX3H0]5:c:c:c:c:[cX3H0]:5:c:[cX3H0]:4:[cX3H0]:3:[cX3H0]:1:2",
    "benzo[a]fluoranthene": "c1:[cX3H0]2:c:c:c:c:[cX3H0]:2:[cX3H0](-[cX3H0]4:c:c:c:c:[cX3H0]5:4):[cX3H0]2:[cX3H0]-5:c:c:c:[cX3H0]:1:2",
    "benzo[b]fluoranthene": "[cX3H0;!$(c12c3cccc1ccc(c24)cccc4cc3)]12:c:c:c:c:[cX3H0]:2:c:[cX3H0](-[cX3H0]4:c:c:c:c:[cX3H0]5:4):[cX3H0]2:[cX3H0]-5:c:c:c:[cX3H0;!$(c12c3cccc1ccc(c24)cccc4cc3)]:1:2",
    "benzo[j]fluoranthene": "c1:c:c:[cX3H0](-[cX3H0]4:[cX3H0]5:c:c:c:c:[cX3H0]:5:c:c:[cX3H0]6:4):[cX3H0]2:[cX3H0]-6:c:c:c:[cX3H0]:1:2",
    "benzo[k]fluoranthene": "c1:c:c:[cX3H0](-[cX3H0]4:c:[cX3H0]5:c:c:c:c:[cX3H0]:5:c:[cX3H0]6:4):[cX3H0]2:[cX3H0]-6:c:c:c:[cX3H0]:1:2",
    "benzo[ghi]fluoranthene": "[cX3H0]12:c:c:c:[cX3H0]3:c:c:[cX3H0]4:c:c:[cX3H0]5:c:c:c:[cX3H0]-2:[cX3H0]:5:[cX3H0]:4:[cX3H0]:3:1",
    "cyclopenta[cd]pyrene": "c1:c:[cX3H0](-[#6;!$(c1ccccc1)]~[#6;!$(c1ccccc1)]5):[cX3H0](:[cX3H0]2:[cX3H0]34):[cX3H0]-5:c:[cX3H0]:4:c:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2",   
    "olympicene": "c15:c:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:[cX3H0]4:c:c:c:c(-C-5):[cX3H0]:4:[cX3H0]:3:[cX3H0]:1:2",
    },
    # 6+ rings
    "Hexacyclic": {
    "hexahelicene": "c1:c:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:[cX3H0]4:c:c:[cX3H0]5:c:c:[cX3H0]6:c:c:c:c:[cX3H0]:6:[cX3H0]:5:[cX3H0]:4:[cX3H0]:3:[cX3H0]:1:2",
    "indeno[cd]pyrene": "c1:c:[cX3H0](-c5:c:c:c:c:c6:5):[cX3H0](:[cX3H0]2:[cX3H0]34):[cX3H0]-6:c:[cX3H0]:4:c:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2", 
    "benzo[ghi]perylene": "[cX3H0]12:c:c:c:[cX3H0]3:c:c:[cX3H0]4:c:c:[cX3H0]5:c:c:[cX3H0]6:c:c:c:[cX3H0]:2:[cX3H0]:6:[cX3H0]:5:[cX3H0]:4:[cX3H0]:1:3",
    "coronene": "[cX3H0]1:c:[cX3H0]2:c:c:[cX3H0]3:c:c:[cX3H0]4:c:c:[cX3H0]5:c:c:[cX3H0]6:c:c:[cX3H0]:1:[cX3H0]7:[cX3H0]:2:[cX3H0]:3:[cX3H0]:4:[cX3H0]:5:[cX3H0]:6:7",
    "corannulene": "c1:c:[cX3H0]2:c:c:[cX3H0]3:c:c:[cX3H0]4:c:c:[cX3H0]5:c:c:[cX3H0]:1:[cX3H0]6:[cX3H0]:2:[cX3H0]:3:[cX3H0]:4:[cX3H0]:5:6",
    }

}

HETEROAROMATICS: Dict[str, str] = {
    # [c;!$(:a)(:a)(:a)] means an aromatic C that is NOT connected to 3 aromatic atoms. 
    # This prevents additional fused aromatic rings.
    "Monocyclic": {
        "5-membered": {
        # 5-membered rings
        ## 1 hetero atomx
        "pyrrole": "[nX3,nX2-;!$([#7]@[BX4]@[#7])]1:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1", # excludes BODIPY
        "N-amino_pyrrole": "[#7;!n]-n1cccc1",
        "N-oxy_pyrrole": "[#8]-n1cccc1",
        "any_pyrrole": "[#7;!X4]1~[#6X3;!$([#6]=O)]~[#6X3;!$([#6]=O)]~[#6X3;!$([#6]=O)]~[#6X3;!$([#6]=O)]~1", 
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
        "1,2-dithiolium": "[s+]1sccc1",
        "1,3-dithiolium": "[s+]1cscc1",
        ## 3 hetero atoms
        "1,2,3-triazole": "n1:n:n:[c;!$(c1ccccc1);!$(c1ncccc1);!$(c1cnccc1);!$(c1ncncc1)]:[c;!$(c1ccccc1);!$(c1ncccc1);!$(c1cnccc1);!$(c1ncncc1)]:1", # either tautomer
        "1,2,4-triazole": "n1:n:[c;!$(c(:a)(:a)(:a));!$(c=[O,S])]:n:[c;!$(c(:a)(:a)(:a));!$(c=[O,S])]:1", # either tautomer
        "1,2,3-oxadiazole": "[oX2]1:[nX2,nX3,nX3+]:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
        "1,2,4-oxadiazole": "[oX2]1:[nX2,nX3,nX3+]:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3,nX3+]:[c;!$(c(:a)(:a)(:a))]:1",
        "1,2,5-oxadiazole": "[oX2]1:[nX2,nX3,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3,nX3+]:1",
        "1,3,4-oxadiazole": "[oX2]1:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3+]:[nX2,nX3,nX3+]:[c;!$(c(:a)(:a)(:a))]:1",
        "1,2,3-thiadiazole": "[sX2]1:[nX2,nX3,nX3+]:[nX2,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
        "1,2,4-thiadiazole": "[sX2]1:[nX2,nX3,nX3+]:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3,nX3+]:[c;!$(c(:a)(:a)(:a))]:1",
        "1,2,5-thiadiazole": "[sX2]1:[nX2,nX3,nX3+]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3,nX3+]:1",
        "1,3,4-thiadiazole": "[sX2]1:[c;!$(c(:a)(:a)(:a))]:[nX2,nX3,nX3+]:[nX2,nX3,nX3+]:[c;!$(c(:a)(:a)(:a))]:1",
        "1,3,2-dithiazolium": "[s+]1nscc1",
        "1,2,5-dithiazolium": "[s+]1sccn1",
        ## 4 hetero atoms
        "tetrazole": "[c;!$(c(:a)(:a)(:a))]1:n:n:n:n:1", # either tautomer
        },
        "6-membered": {
        # 6-membered rings
        ## 1 hetero atom
        "pyridine": "[nX2,nX3H+]1:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
        "pyridinium": "[nX3H0+]1:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:[c;!$(c(:a)(:a)(:a))]:1",
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
        "1,3,5-triazine": "[nX2,nX3+]1:[c;!$(c1(~[O,N])nc(~[O,N])nc(~[O,N])n1)]:[nX2,nX3+]:[c;!$(c1(~[O,N])nc(~[O,N])nc(~[O,N])n1)]:[nX2,nX3+]:[c;!$(c1(~[O,N])nc(~[O,N])nc(~[O,N])n1)]:1",
        "melamine": "N~c1:n:c(~N):n:c(~N):n:1",

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
        },
        "7-membered": {
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
        }
    },

    "Bicyclic": {
        "5-5": {
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
        },
        "5-6": {
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
        ## 4 hetero atoms +
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
        ### (1@bridge triazine-based)
        "imidazo[1,2-a]-1,3,5-triazine": "[nX2,nX3+]1:c:c:n2:c:[nX2,nX3+]:c:[nX2,nX3+]:c:1:2",
        "imidazo[1,5-a]-1,3,5-triazine": "c1:[nX2,nX3+]:c:n2:c:[nX2,nX3+]:c:[nX2,nX3+]:c:1:2",
        "pyrazolo[1,5-a]-1,3,5-triazine": "c1:c:[nX2,nX3+]:n2:c:[nX2,nX3+]:c:[nX2,nX3+]:c:1:2",
        "5-aza-purine": "[nX2,nX3+]1:c:[nX2,nX3+]:n2:c:[nX2,nX3+]:c:[nX2,nX3+]:c:1:2",
        ### misc
        "triazolopyrimidine": "[nX3,nX2-]1:[nX2,nX3+]:[nX2,nX3+]:[cX3H0]2:[nX2,nX3,nX3+]:c:[nX2,nX3,nX3+]:c:[cX3H0]:1:2",
        },
        "6-6": {
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
        }
    },
    
    "Tricyclic": {
        "5-5-6": {
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
        },
        "5-6-5": {
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
        "furo[2,3-e]indolizine": "o1:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:c:[nX3]:3:[cX3H0]:1:2",
        "furo[3,2-e]indolizine": "c1:c:o:[cX3H0]2:c:c:[cX3H0]3:c:c:c:[nX3]:3:[cX3H0]:1:2",
        "furo[3,4-e]indolizine": "c1:o:c:[cX3H0]2:c:c:[cX3H0]3:c:c:c:[nX3]:3:[cX3H0]:1:2",
        "furo[2,3-f]indolizine": "o1:c:c:[cX3H0]2:c:[cX3H0]3:c:c:c:[nX3]:3:c:[cX3H0]:1:2",
        "furo[3,2-f]indolizine": "c1:c:o:[cX3H0]2:c:[cX3H0]3:c:c:c:[nX3]:3:c:[cX3H0]:1:2",
        "furo[2,3-g]indolizine": "o1:c:c:[cX3H0]2:[cX3H0]3:c:c:c:[nX3]:3:c:c:[cX3H0]:1:2",
        "furo[3,2-g]indolizine": "c1:c:o:[cX3H0]2:[cX3H0]3:c:c:c:[nX3]:3:c:c:[cX3H0]:1:2",
        "furo[3,4-g]indolizine": "c1:o:c:[cX3H0]2:[cX3H0]3:c:c:c:[nX3]:3:c:c:[cX3H0]:1:2",
        "thieno[2,3-e]indolizine": "[sX2]1:c:c:[cX3H0]2:c:c:[cX3H0]3:c:c:c:[nX3]:3:[cX3H0]:1:2",
        "thieno[3,2-e]indolizine": "c1:c:[sX2]:[cX3H0]2:c:c:[cX3H0]3:c:c:c:[nX3]:3:[cX3H0]:1:2",
        "thieno[3,4-e]indolizine": "c1:[sX2]:c:[cX3H0]2:c:c:[cX3H0]3:c:c:c:[nX3]:3:[cX3H0]:1:2",
        "thieno[2,3-f]indolizine": "[sX2]1:c:c:[cX3H0]2:c:[cX3H0]3:c:c:c:[nX3]:3:c:[cX3H0]:1:2",
        "thieno[3,2-f]indolizine": "c1:c:[sX2]:[cX3H0]2:c:[cX3H0]3:c:c:c:[nX3]:3:c:[cX3H0]:1:2",
        "thieno[2,3-g]indolizine": "[sX2]1:c:c:[cX3H0]2:[cX3H0]3:c:c:c:[nX3]:3:c:c:[cX3H0]:1:2",
        "thieno[3,2-g]indolizine": "c1:c:[sX2]:[cX3H0]2:[cX3H0]3:c:c:c:[nX3]:3:c:c:[cX3H0]:1:2",
        "thieno[3,4-g]indolizine": "c1:[sX2]:c:[cX3H0]2:[cX3H0]3:c:c:c:[nX3]:3:c:c:[cX3H0]:1:2",
        },
        "6-5-6": {
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
        },
        "6-6-6": {
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
        },
    },

    "Azapyrene": {
        # azapyrenes
        "1-aza-pyrene": "n1:c:c:[cX3H0](:[cX3H0]2:[cX3H0]34):c:c:[cX3H0]:4:c:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2",  
        "2-aza-pyrene": "c1:n:c:[cX3H0](:[cX3H0]2:[cX3H0]34):c:c:[cX3H0]:4:c:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2",  
        "4-aza-pyrene": "c1:c:c:[cX3H0](:[cX3H0]2:[cX3H0]34):n:c:[cX3H0]:4:c:c:c:[cX3H0]:3:c:c:[cX3H0]:1:2", 
        "2,7-diaza-pyrene": "c1:n:c:[cX3H0](:[cX3H0]2:[cX3H0]34):c:c:[cX3H0]:4:c:n:c:[cX3H0]:3:c:c:[cX3H0]:1:2",  
    },

    "Cyclazine": {
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
    },

    "Porphyrinoid": {
        "porphyrin": "[#6;X3H0]12~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~2)~[#6;X3]~[#6;X3H0]3~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~3)~[#6;X3]~[#6;X3H0]4~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~4)~[#6;X3]~[#6;X3H0]5~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~5)~[#6;X3]~1",
        "chlorin": "[#6;X3H0]12-C-C-[#6;X3H0](~[#7]~2)~[#6;X3]~[#6;X3H0]3~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~3)~[#6;X3]~[#6;X3H0]4~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~4)~[#6;X3]~[#6;X3H0]5~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~5)~[#6;X3]~1",
        "bacteriochlorin": "[#6;X3H0]12-C-C-[#6;X3H0](~[#7]~2)~[#6;X3]~[#6;X3H0]3~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~3)~[#6;X3]~[#6;X3H0]4-C-C-[#6;X3H0](~[#7]~4)~[#6;X3]~[#6;X3H0]5~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~5)~[#6;X3]~1",
        "isobacteriochlorin": "[#6;X3H0]12-C-C-[#6;X3H0](~[#7]~2)~[#6;X3]~[#6;X3H0]3-C-C-[#6;X3H0](~[#7]~3)~[#6;X3]~[#6;X3H0]4~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~4)~[#6;X3]~[#6;X3H0]5~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~5)~[#6;X3]~1",
        "corrole": "[#6;X3H0]12~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~2)~[#6;X3H0]3~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~3)~[#6;X3]~[#6;X3H0]4~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~4)~[#6;X3]~[#6;X3H0]5~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~5)~[#6;X3]~1",
        "norcorrole": "[#6;X3H0]12~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~2)~[#6;X3H0]3~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~3)~[#6;X3]~[#6;X3H0]4~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~4)~[#6;X3H0]5~[#6;X3]=,:[#6;X3]~[#6;X3H0](~[#7]~5)~[#6;X3]~1",
        "corrin": "[#6;X3H0]12-C-C-C(~[#7]~2)-C3-C-C-[#6;X3H0](~[#7]~3)~[#6;X3]~[#6;X3H0]4-C-C-[#6;X3H0](~[#7]~4)~[#6;X3]~[#6;X3H0]5-C-C-[#6;X3H0](~[#7]~5)~[#6;X3]~1",
        "porphycene": "[#6;X3H0]12~[#6]~[#6]~[#6;X3H0](~[#7]~2)~[#6;X3]~[#6;X3]~[#6;X3H0]3~[#6]~[#6]~[#6;X3H0](~[#7]~3)~[#6;X3H0]4~[#6]~[#6]~[#6;X3H0](~[#7]~4)~[#6;X3]~[#6;X3]~[#6;X3H0]5~[#6]~[#6]~[#6;X3H0](~[#7]~5)~1", # including partial saturations,
        "porphyrazine": "[#6;X3H0]12~[#6]~[#6]~[#6;X3H0](~[#7]~2)~[#7;X2,X3+]~[#6;X3H0]3~[#6]~[#6]~[#6;X3H0](~[#7]~3)~[#7;X2,X3+]~[#6;X3H0]4~[#6]~[#6]~[#6;X3H0](~[#7]~4)~[#7;X2,X3+]~[#6;X3H0]5~[#6]~[#6]~[#6;X3H0](~[#7]~5)~[#7;X2,X3+]~1", # includin phthalocyanin
        },
} 


ALIPHATIC_RINGS: Dict[str, str]= {
    "Monocyclic": {
        "Saturated": {
            # saturated
            "cyclopropane": "C1-C-C-1",
            "cyclobutane":  "[C;!$(C12C3C4C1C5C2C3C45);!$([C;R2]12CC@2C1)]1-[C;!$(C12C3C4C1C5C2C3C45);!$([C;R2]12CC@2C1)]-[C;!$(C12C3C4C1C5C2C3C45);!$([C;R2]12CC@2C1)]-[C;!$(C12C3C4C1C5C2C3C45);!$([C;R2]12CC@2C1)]-1", # excludes cubane
            "cyclopentane": "[C;!$([C;R2]12CC@2CC1)]1-[C;!$([C;R2]12CC@2CC1)]-[C;!$([C;R2]12CC@2CC1)]-[C;!$([C;R2]12CC@2CC1)]-[C;!$([C;R2]12CC@2CC1)]-1",
            "cyclohexane":  "[C;!$(C12CCC(CC1)CC2);!$(C12C3C4C1C5C2C3C45);!$([C;R2]12CC@2CCC1);!$([C;R2]12CCC@2CC1)]1-[C;!$(C12CCC(CC1)CC2);!$(C12C3C4C1C5C2C3C45);!$([C;R2]12CC@2CCC1);!$([C;R2]12CCC@2CC1)]-[C;!$(C12CCC(CC1)CC2);!$(C12C3C4C1C5C2C3C45);!$([C;R2]12CC@2CCC1);!$([C;R2]12CCC@2CC1)]-[C;!$(C12CCC(CC1)CC2);!$(C12C3C4C1C5C2C3C45);!$([C;R2]12CC@2CCC1);!$([C;R2]12CCC@2CC1)]-[C;!$(C12CCC(CC1)CC2);!$(C12C3C4C1C5C2C3C45);!$([C;R2]12CC@2CCC1);!$([C;R2]12CCC@2CC1)]-[C;!$(C12CCC(CC1)CC2);!$(C12C3C4C1C5C2C3C45);!$([C;R2]12CC@2CCC1);!$([C;R2]12CCC@2CC1)]1", # excludes cubane
            "cycloheptane": "[C;!$([C;R2]12CC@2CCCC1);!$([C;R2]12CCC@2CCC1)]1-[C;!$([C;R2]12CC@2CCCC1);!$([C;R2]12CCC@2CCC1)]-[C;!$([C;R2]12CC@2CCCC1);!$([C;R2]12CCC@2CCC1)]-[C;!$([C;R2]12CC@2CCCC1);!$([C;R2]12CCC@2CCC1)]-[C;!$([C;R2]12CC@2CCCC1);!$([C;R2]12CCC@2CCC1)]-[C;!$([C;R2]12CC@2CCCC1);!$([C;R2]12CCC@2CCC1)]-[C;!$([C;R2]12CC@2CCCC1);!$([C;R2]12CCC@2CCC1)]-1",
            "cyclooctane":  "[C;!$(C12C3C4C1C5C2C3C45);!$([C;R2]12CC@2CCCCC1);!$([C;R2]12CCC@2CCCC1);!$([C;R2]12CCCC@2CCC1)]1-[C;!$(C12C3C4C1C5C2C3C45);!$([C;R2]12CC@2CCCCC1);!$([C;R2]12CCC@2CCCC1);!$([C;R2]12CCCC@2CCC1)]-[C;!$(C12C3C4C1C5C2C3C45);!$([C;R2]12CC@2CCCCC1);!$([C;R2]12CCC@2CCCC1);!$([C;R2]12CCCC@2CCC1)]-[C;!$(C12C3C4C1C5C2C3C45);!$([C;R2]12CC@2CCCCC1);!$([C;R2]12CCC@2CCCC1);!$([C;R2]12CCCC@2CCC1)]-[C;!$(C12C3C4C1C5C2C3C45);!$([C;R2]12CC@2CCCCC1);!$([C;R2]12CCC@2CCCC1);!$([C;R2]12CCCC@2CCC1)]-[C;!$(C12C3C4C1C5C2C3C45);!$([C;R2]12CC@2CCCCC1);!$([C;R2]12CCC@2CCCC1);!$([C;R2]12CCCC@2CCC1)]-[C;!$(C12C3C4C1C5C2C3C45);!$([C;R2]12CC@2CCCCC1);!$([C;R2]12CCC@2CCCC1);!$([C;R2]12CCCC@2CCC1)]-[C;!$(C12C3C4C1C5C2C3C45);!$([C;R2]12CC@2CCCCC1);!$([C;R2]12CCC@2CCCC1);!$([C;R2]12CCCC@2CCC1)]-1", # excludes cubane
        },
        "Unsaturated": {
            # mono unsaturated
            "cyclopropene":  "[CX3]1=[CX3]-C-1",
            "cyclobutene":   "[#6;!$([C;R2]12@C@C@2@C@1)]1=,:[#6;!$([C;R2]12@C@C@2@C@1)]-[C;!$([C;R2]12@C@C@2@C@1)]-[C;!$([C;R2]12@C@C@2@C@1)]-1",
            "cyclopentene":  "[C;!$([C;R2]12@C@C@2@C@C@1)]1=[C;!$([C;R2]12@C@C@2@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@1)]-1",
            "cyclohexene":   "[C;!$([C;R2]12@C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]1=[C;!$([C;R2]12@C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@1)]-1",
            "cycloheptene":  "[#6;X3;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]1=,:[#6;X3;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@1)]-1",
            "cyclooctene":   "[#6;X3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]1=,:[#6;X3;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-1",
            ## cyclooctynes and related
            "cyclooctyne":   "[CX2]1#[CX2]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-1",
            "benzocyclooctyne": "[CX2]1#[CX2]-c2ccccc2-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-[C;!$([C;R2]12@C@C@2@C@C@C@C@C@1);!$([C;R2]12@C@C@C@2@C@C@C@C@1);!$([C;R2]12@C@C@C@C@2@C@C@C@1)]-1",
            "dibenzocyclooctyne": "[CX2]1#[CX2]-c2ccccc2-CC-c2ccccc21",
            'aza-dibenzocyclooctyne': "[CX2]1#[CX2]-c2ccccc2-NC-c2ccccc21",
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
        }
    },
        "Polycyclic": {
            # bicyclic
            "spiro_carbon": "[CD4;R2;x4]",
            "indane": "c1:c:c:c:[cX3H0]2-[C;!$([C;R2]12CC@2cc1)]-[C;!$([C;R2]12CC@2cc1)]-[C;!$([C;R2]12CC@2cc1)]-[cX3H0]:1:2",
            "tetralin": "c1:c:c:c:[cX3H0]2-[C;!$(C12-cc-C2CC1);!$(C12-cc-CC2C1);!$(C1-cc-C2CC21)]CCC-[cX3H0]:1:2",
            "decalin": "[C;!$(C123CC3CCC2CCCC1);!$(C123CCC3CC2CCCC1);!$(C123CCCC3C2CCCC1);!$(C12C3CC3CC2CCCC1);!$(C12C3CCC3C2CCCC1);!$(C12C3CCCC23CCCC1)]12CCCC[C;!$(C123CC3CCC2CCCC1);!$(C123CCC3CC2CCCC1);!$(C123CCCC3C2CCCC1);!$(C12C3CC3CC2CCCC1);!$(C12C3CCC3C2CCCC1);!$(C12C3CCCC23CCCC1)]2CCCC1",
            "norbornane": "C12CCC(C1)CC2",
            "norbornene": "C12[#6;X3]=,:[#6;X3]C(C1)CC2",
            "norbornadiene": "C12[#6;X3]=,:[#6;X3]C(C1)[#6;X3]=,:[#6;X3]2",
            "norpinane": "[C;!$(C12C3C4C1C5C2C3C45)]12CCC[C;!$(C12C3C4C1C5C2C3C45)]([C;!$(C12C3C4C1C5C2C3C45)]1)[C;!$(C12C3C4C1C5C2C3C45)]2", # excludes cubane
            "bicyclo[2.2.2]octane": "[C;!$(C12CCC(CC13)CC32)]12CC[C;!$(C12CCC(CC13)CC32)](CC1)CC2",

            # 3+ rings
            "cubane": 'C12C3C4C1C5C2C3C45',
            "adamantane": "C12CC(C3)CC(C2)CC3C1",
            "steroid_rings": "[#6]1~[#6]~[#6]~[#6]~[#6]2~[#6]~[#6]~[#6]3~[#6]4~[#6]~[#6]~[#6]~[#6]~4~[#6]~[#6]~[#6]~3~[#6]~2~1",
    }

}

HETEROALIPHATIC_RINGS: Dict[str, str] = {
    # !$(C=[O,S]) next to heteroatoms prevents OXO / THIOXO variants
    # !$(C=[N,O,S]) is used next to nitrogens to also exclude amidines and guanidines
    "3-Membered": {
        # 3-membered
        "epoxide": "O1[CX4][CX4]1",
        "aziridine": "N1[CX4][CX4]1",
        "azirine": "N1=[CX3][CX4]1",
        "episulfide": "[SX2]1[CX4][CX4]1",
        "episelenide": "[Se]1[CX4][CX4]1",
        "oxaziridine": "O1N[CX4]1",
    },
    "4-Membered": {
        # 4-membered
        "azetidine": "N1-[C;!$(C=O)]-C-[C;!$(C=O)]-1", # excludes beta-lactams
        "azetine": "N1[#6;X3]=,:[#6;X3]-C-1",
        "oxetane": "O1-[C;!$(C=[O,S])]-C-[C;!$(C=[O,S])]-1", # excludes beta-lactones
        "oxetine": "O1[#6;X3]=,:[#6;X3]-C-1",
        "1,3-dioxetane": "O1-C-O-C-1",
        "thietane": "S1-[C;!$(C=[O,S])]-C-[C;!$(C=[O,S])]-1", # excludes beta-thiolactones
        "1,2-dithietane": "S1-S-C-C-1",
        "1,3-dithietane": "S1-C-S-C-1",
    },
    "5-Membered": {
        # 5-membered
        "pyrrolidine": "[N;!$(N12[#6]~[#6]~[#6]C2~[#6]~[#6]~[#6]~1);!$(N12[#6]~[#6]~[#6]C2~[#6]~[#6]~[#6]~[#6]~1)]1-[C;!$(C=[N,O,S])]-C-C-[C;!$(C=[N,O,S])]-1", # excludes gamma-lactams, pyrrolizidine, indolizidine
        "1-pyrroline": "N1=[CX3]-C-C-[C;!$(C=O)]-1",
        "2-pyrroline": "N1-[CX3]=[CX3]-C-[C;!$(C=O)]-1", # excludes indoline
        "3-pyrroline": "N1-[C;!$(C=O)]-[CX3]=[CX3]-[C;!$(C=O)]-1",
        "pyrazolidine": "N1-N-[C;!$(C=O)]-C-[C;!$(C=O)]-1",
        "1-pyrazoline": "[NX2,NX3+]1=[NX2,NX3+]-[C;!$(C=O)]-C-[C;!$(C=O)]-1",
        "2-pyrazoline": "N1-[NX2,NX3+]=[CX3]-C-[C;!$(C=O)]-1",
        "imidazolidine": "N1-[C;!$(C=O)]-N-C-[C;!$(C=O)]-1",
        "3-imidazoline": "N1-[C;!$(C=O)]-[NX2,NX3+]=[CX3]-[C;!$(C=O)]-1",
        "4-imidazoline": "N1-[CX4;!$(C=O)]-N-[CX3]=[CX3]-1",
        "oxolane": "O1-[C;!$(C=[O,S])]-C-C-[C;!$(C=[O,S])]-1", # excludes gamma-lactones
        "benzo[b]oxolane": "O1-c2ccccc2-[CX4]-[CX4]-1",
        "benzo[c]oxolane": "O1-[CX4]-c2ccccc2-[CX4]-1",
        "2-oxolene": "O1-[CX3]=[CX3]-C-[C;!$(C=[O,S])]-1",
        "3-oxolene": "O1-[C;!$(C=[O,S])]-[CX3]=[CX3]-[C;!$(C=[O,S])]-1",
        "thiolane": "[SX2]1-[C;!$(C=[O,S])]-C-C-[C;!$(C=[O,S])]-1", # excludes gamma-thiolactones
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
        "1,3-dithiole": "[SX2;!$([SX2]1-[#6;X3]=,:[#6;X3]-[SX2]-C-1=C2-[SX2]-[#6;X3]=,:[#6;X3]-[SX2]-2)]1-[C;!$(C=[O,S])]-[SX2]-[CX3]=[CX3]-1", # excludes tetrathiafulvalene
        "1,3-oxathiolane": "[SX2]1-[C;!$(C=[O,S])]-O-C-[C;!$(C=[O,S])]-1",
        "1,3-oxathiole": "[SX2]1-[C;!$(C=[O,S])]-O-[CX3]=[CX3]-1",
        "1,3-oxazolidine": "N1-[C;!$(C=[O,S])]-O-C-[C;!$(C=[O,S])]-1",
        "2-oxazoline": "[NX2,NX3+]1=[CX3]-O-[C;!$(C=[O,S])]-[C;!$(C=[O,S])]-1",
        "3-oxazoline": "N1-[C;!$(C=[O,S])]-O-[C;!$(C=[O,S])]-[CX3]=1",
        "4-oxazoline": "N1-[C;!$(C=[O,S])]-O-[CX3]=[CX3]-1",
    },
    "6-Membered": {
        # 6-membered
        "piperidine": "[#7;!$(N12C[#6]~[#6]([#6]~[#6]2)[#6]~[#6]1);!$(N12[#6]~[#6]~[#6]~[#6]C2~[#6]~[#6]~[#6]~1);!$(N12[#6]~[#6]~[#6]~[#6]C2~[#6]~[#6]~[#6]~[#6]~1);!$([#7]1-C2-[#6]~[#6]-C(-[#6]~1)-[#6]~[#6]-2)]1-,:[#6;!$(C=[O,S])]-C-C-C-[#6;!$(C=[O,S])]-,:1", # excludes delta-lactams, squinuclidine, isoquinuclidine, indolizidine, quinolizidine
        "piperazine": "[#7;!$([#7]1C(=O)[#6][#7]C(=O)[#6]1)]1-,:[#6]-,:[#6]-[N;!$([#7]1C(=O)[#6][#7]C(=O)[#6]1);!$(N12-[#6]~[#6]-N(-[#6]~[#6]-2)-[#6]~[#6]-1)]-C-C-1", # excludes diketopiperazine
        "1,4-dihydropyridine": "[#7]1-[#6]=,:[#6]-[CX4]-[#6]=,:[#6]-1",
        "oxane": "O1-[C;!$(C=[O,S])]-C-C-C-[C;!$(C=[O,S])]-1", # excludes delta-lactones
        "1,2-dioxane": "O1-O-[C;!$(C=[O,S])]-C-C-[C;!$(C=[O,S])]-1",
        "1,3-dioxane": "O1-[C;!$(C=[O,S])]-O-[C;!$(C=[O,S])]-C-[C;!$(C=[O,S])]-1",
        "1,4-dioxane": "O1-[C;!$(C=[O,S])]-[C;!$(C=[O,S])]-O-[C;!$(C=[O,S])]-[C;!$(C=[O,S])]-1",
        "thiane": "[SX2]1-[C;!$(C=[O,S])]-C-C-C-[C;!$(C=[O,S])]-1", # excludes delta-thiolactones
        "1,2-dithiane": "[SX2]1[SX2]-[C;!$(C=[O,S])]-C-C-[C;!$(C=[O,S])]-1",
        "1,3-dithiane": "[SX2]1-[C;!$(C=[O,S])]-[SX2]-[C;!$(C=[O,S])]-C-[C;!$(C=[O,S])]-1",
        "1,4-dithiane": "[SX2]1-[C;!$(C=[O,S])]-[C;!$(C=[O,S])]-[SX2]-[C;!$(C=[O,S])]-[C;!$(C=[O,S])]-1",
        "morpholine": "O1-[#6]-,:[#6]-[#7]-,:[#6]-,:[#6]-1",
        "thiomorpholine": "[SX2]1-[#6]-,:[#6]-[#7]-,:[#6]-,:[#6]-1",
        "chromane": "O1CCCc2ccccc12",
        "isochromane": "C1OCCc2ccccc12",
        "thiochromane": "S1CCCc2ccccc12",
        "isothiochromane": "C1SCCc2ccccc12",
        "DABCO": "N12-[#6]-,=,:[#6]-N(-[#6]-,=,:[#6]-2)-[#6]-,=,:[#6]-1",    
    }
}

OXO_RINGS: Dict[str, str] = {

    "3-Membered": {
        # 3-membered
        "cyclopropenone": "O=[c;!$(c1c([O,N])c([O,N])1)]1cc1",
        "α-lactone": "O=C1-O-[#6;!$(C=[O,S])]-1",
        "α-lactam": "O=C1-N-[#6;!$(C=[O,S])]-1",
        "α-thiolactone": "O=C1-[SX2]-[#6;!$(C=[O,S])]-1",
    },
    "4-Membered": {  
        # 4-membered
        "cyclobutenedione": "O=[c;!$(c1cc([O,N,S])c([O,N,S])1)]1c(=O)cc1",
        "β-lactone": "O=C1-O-[#6;!$(C=[O,S])]~[#6]-1",
        "β-lactam": "O=C1-N-[#6;!$(C=[O,S])]~[#6]-1",
        "β-thiolactone": "O=C1-[SX2]-[#6;!$(C=[O,S])]~[#6]-1",
    },
    "5-Membered": {
        # 5-membered
        "γ-lactone": "O=C1-O-[#6;!$(C=[O,S])]~[#6]-[#6]-1", # excludes butenolide
        "butenolide": "O=C1-O-[#6;!$(C=[O,S])]~[#6]=,:[#6]-1",
        "γ-lactam": "O=C1-N-[#6;!$(C=[O,S])]~[#6]~[#6]-1",
        "γ-thiolactone": "O=C1-[SX2]-[#6;!$(C=[O,S])]~[#6]~[#6]-1",
        "tetronic_acid": "O=C1-O-[C;!$(C=[O,S,N])]-C(-O)=C-1",
        "tetramic_acid": "O=C1-N-[C;!$(C=[O,S,N])]-C(-O)=C-1",
        "cyclopentadienone": "O=C1C=CC=C1",
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
        "oxazolone": "O=c1occn1",
        "2-oxazolidinone": "O=C1N[CX4][CX4]O1",
        "2-thiazolone": "O=c1[sX2]ccn1",
        "isoxazolinone": "O=c1nocc1",
        "isothiazolinone": "O=c1n[sX2]cc1",
        "isoxazolidinone": "O=C1NOCC1",
        "isothiazolidinone": "O=C1N[SX2]CC1",
        "oxazolidinedione": "O=C1NC(=O)[CX4]O1",
        "2-thiazolidinone": "O=C1N[CX4][CX4][SX2]1",
        "4-thiazolidinone": "O=C1N[CX4][SX2][CX4]1",
        "thiazolidindione": "O=C1NC(=O)[CX4][SX2]1",
        "1,2,4-triazolinone": "O=c1ncnn1",
        "1,2,4-triazolidinone": "O=C1NCNN1",
        "1,2,4-triazolidinedione": "O=c1nc(=O)nn1",
    },
    "6&7-Membered": {
        # 6-membered
        "δ-lactone": "O=C1-O-[#6;!$(C=[O,S])]~[#6]~[#6]~[#6]-1",
        "δ-lactam": "O=C1-N-[#6;!$(C=[O,S])]~[#6]~[#6]~[#6]-1",
        "δ-thiolactone": "O=C1-[SX2]-[#6;!$(C=[O,S])]~[#6]~[#6]~[#6]-1",
        "1,2-benzoquinone": "O=C1-C(=O)-[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-1",
        "1,4-benzoquinone": "O=C1-[#6;X3]=,:[#6;X3]-C(=O)-[#6;X3]=,:[#6;X3]-1",
        "1,2-quinone_methide": "[O;!$(O=C1-[#6;X3]=,:[#6;X3]-,:[#6;X3]=[#6;X3]2-[#6;X3]=,:[#6;X3]-C(=O)-[#6;X3]=[#6;X3]12);!$(O=C1-[#6;X3]=,:[#6;X3]-,:[#6;X3]=[#6;X3]2-C(=O)-[#6;X3]=,:[#6;X3]-,:[#6;X3]=[#6;X3]12)]=C1-C(=C)-[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-1", # excludes naphthoquinones
        "1,2-quinone_dimethide": "C=c1c(=C)cccc1",
        "1,4-quinone_methide": "[O;!$(O=C1-[#6;X3]=,:[#6;X3]-[#6;X3]2=[#6;X3]-,:[#6;X3]=,:[#6;X3]-C(=O)-[#6;X3]2=[#6;X3]1);!$(O=C1-[#6;X3]=,:[#6;X3]-[#6;X3]2=[#6;X3]-C(=O)-[#6;X3]=,:[#6;X3]-[#6;X3]-2=[#6;X3]-1)]=C1-[#6;X3]=,:[#6;X3]-C(=C)-[#6;X3]=,:[#6;X3]-1", # excludes naphthoquinones
        "1,4-quinone_dimethide": "C=c1ccc(=C)cc1",
        "1,5-naphthoquinone": "O=C1-[#6;X3]=,:[#6;X3]-,:[#6;X3]=[#6;X3]2-C(=O)-[#6;X3]=,:[#6;X3]-,:[#6;X3]=[#6;X3]12",
        "1,7-naphthoquinone": "O=C1-[#6;X3]=,:[#6;X3]-,:[#6;X3]=[#6;X3]2-[#6;X3]=,:[#6;X3]-C(=O)-[#6;X3]=[#6;X3]12",
        "2,6-naphthoquinone": "O=C1-[#6;X3]=,:[#6;X3]-[#6;X3]2=[#6;X3]-C(=O)-[#6;X3]=,:[#6;X3]-[#6;X3]-2=[#6;X3]-1",
        "2-pyrone": "O=c1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:o1", # excludes benzopyrone
        "thiopyran-2-one": "O=c1:c:c:c:c:s1",
        "4-pyrone": "O=c1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:o:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]1", # excludes benzopyrone
        "thiopyran-4-one": "O=c1:c:c:s:c:c1",
        "1-benzo[c]pyrone": "O=c1:o:c:c:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
        "2-benzo[b]pyrone": "o1:c(=O):c:c:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
        "3-benzo[c]pyrone": "c1:o:c(=O):c:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
        "4-benzo[b]pyrone": "o1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:c(=O):[cX3H0]2:c:c:c:c:[cX3H0]:1:2", # excludes xanthone
        "xanthone": "o1:[cX3H0]2:c:c:c:c:[cX3H0]:2:c(=O):[cX3H0]3:c:c:c:c:[cX3H0]:3:1",
        "2-pyridone": "O=c1:[n;!$(n12c(=O)cccc1cccc2);!$(n12c(=O)cccc1nccc2);!$(n12c(=O)cccc1cncc2);!$(n12c(=O)cccc1ccnc2);!$(n12c(=O)cccc1cccn2)]:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:1", # excludes quinolones, quinolizinones
        "4-pyridone": "O=c1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[n;!$(n12ccc(=O)cc1cccc2);!$(n12ccc(=O)cc1nccc2);!$(n12ccc(=O)cc1cncc2);!$(n12ccc(=O)cc1ccnc2);!$(n12ccc(=O)cc1cccn2)]:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:1", # excludes quinolones, quinolizinones
        "1-isoquinolone": "O=c1:n:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
        "2-quinolone":    "n1:c(=O):[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
        "3-isoquinolone": "c1:n:c(=O):c:[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
        "4-quinolone":    "n1:[c;!$(c1ccccc1)]:[c;!$(c1ccccc1)]:c(=O):[cX3H0]2:c:c:c:c:[cX3H0]:1:2",
        "acridone":       "n1:[cX3H0]2:c:c:c:c:[cX3H0]:2:c(=O):[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
        "phenanthridone": "n1:c(=O):[cX3H0]2:c:c:c:c:[cX3H0]:2:[cX3H0]3:c:c:c:c:[cX3H0]:1:3",
        "2-pyrimidone": "O=c1:[n;!$(n12c(=O)nccc1cccc2)]:[c;!$(c~[O,N]);!$(c1ccccc1)]:[c;!$(c1ncnc1);!$(c1ccccc1)]:[c;!$(c~[O,N]);!$(c1ncnc1);!$(c1ccccc1)]:[n;!$(n12c(=O)nccc1cccc2)]:1", # excludes nucleobases, quinazolin-2-one, 3-aza-quinolizin-4-one
        "4-pyrimidone": "O=c1:[n;!$(n12c(=O)ccnc1cccc2)]:[c;!$(c~[O,N])]:[n;!$(n12ccc(=O)nc1cccc2);!$(n12cnc(=O)cc1cccc2)]:[c;!$(c1ncnc1);!$(c1ccccc1)]:[c;!$(c1ncnc1);!$(c1ccccc1)]:1", # excludes nucleobases, quinazolin-4-one, aza-quinolizinones
        "pyridazin-3-one": "O=c1:n:n:[c;!$(c=O)]:c:c:1", # excludes pyridazinedione
        "pyridazin-5-one": "O=c1:c:n:[n;!$(n12ncc(=O)cc1cccc2)]:c:c:1",
        "pyridazine-3,6-dione": "O=c1:n:n:c(=O):c:c:1",
        "pyrazinone": "O=c1:[n;!$(n12c(=O)cncc1cccc2)]:c:c:n:[c;!$(c=O)]:1", # excludes pyrazinedione
        "pyrazine-2,3-dione": "O=c1:n:c:c:n:c(=O):1", 
        "2,5-diketopiperazine": "O=C1[#7]-,:[#6]C(=O)[#7]-,:[#6]1",
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
        "barbiturate": "[OX1,OH]~[#6;X3H0]1~N~[#6;X3H0](~[OX1,OH])~N~[#6;X3H0](~[OX1,OH])-C-1",
        "quinazolin-2-one":  "O=c1:n:c:c2ccccc:2:n:1",
        "quinazolin-4-one":  "O=c1:n:c:n:c2ccccc:1:2",
        "ammelide": "O~c1:n:c(~O):n:c(~N):n:1",
        "ammeline": "O~c1:n:c(~N):n:c(~N):n:1",
        "cyanuric_acid": "O~c1:n:c(~O):n:c(~O):n:1",
         # 7-membered
        "tropone": "[OX1]~c1:[c;!$(c~O)]:c:c:c:c:[c;!$(c~O)]:1", # excludes tropolone
        "tropolone": "O~c1:c(O):c:c:c:c:c:1"
    },
    "Quinolizine": {
        # quinolizins
        "quinolizin-2-one": "c12:c:c(=O):c:c:n:1:c:c:c:c:2",
        "quinolizin-4-one": "c12:c:c:c:c(=O):n:1:c:c:c:c:2",
        "1-aza-quinolizin-2-one": "c12:n:c(=O):c:c:n:1:c:c:c:c:2",
        "1-aza-quinolizin-4-one": "c12:n:c:c:c(=O):n:1:c:c:c:c:2",
        "1-aza-quinolizin-6-one": "c12:n:c:c:c:n:1:c(=O):c:c:c:2",
        "1-aza-quinolizin-8-one": "c12:n:c:c:c:n:1:c:c:c(=O):c:2",
        "2-aza-quinolizin-4-one": "c12:c:n:c:c(=O):n:1:c:c:c:c:2",
        "2-aza-quinolizin-6-one": "c12:c:n:c:c:n:1:c(=O):c:c:c:2",
        "2-aza-quinolizin-8-one": "c12:c:n:c:c:n:1:c:c:c(=O):c:2",
        "3-aza-quinolizin-2-one": "c12:c:c(=O):n:c:n:1:c:c:c:c:2",
        "3-aza-quinolizin-4-one": "c12:c:c:n:c(=O):n:1:c:c:c:c:2",
        "3-aza-quinolizin-6-one": "c12:c:c:n:c:n:1:c(=O):c:c:c:2",
        "3-aza-quinolizin-8-one": "c12:c:c:n:c:n:1:c:c:c(=O):c:2",
        "4-aza-quinolizin-2-one": "c12:c:c(=O):c:n:n:1:c:c:c:c:2",
        "4-aza-quinolizin-6-one": "c12:c:c:c:n:n:1:c(=O):c:c:c:2",
        "4-aza-quinolizin-8-one": "c12:c:c:c:n:n:1:c:c:c(=O):c:2",
    },
    "Pteridine": {
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
    },
}

THIOXO_RINGS: Dict[str, str] = {

    "3-Membered": {
        # 3-membered
        "cyclopropenthione": "[SX1]=[c;!$(c1c([O,N])c([O,N])1)]1cc1",
        "α-thionolactone": "[SX1]=C1-O-[#6;!$(C=[O,S])]-1",
        "α-thiolactam": "[SX1]=C1-N-[#6;!$(C=[O,S])]-1",
        "α-dithiolactone": "[SX1]=C1-[SX2]-[#6;!$(C=[O,S])]-1",
    },
    "4-Membered": {
        # 4-membered
        "β-thionolactone": "[SX1]=C1-O-[#6;!$(C=[O,S])]~[#6]-1",
        "β-thiolactam": "[SX1]=C1-N-[#6;!$(C=[O,S])]~[#6]-1",
        "β-dithiolactone": "[SX1]=C1-[SX2]-[#6;!$(C=[O,S])]~[#6]-1",
    },
    "5-Membered": {
        # 5-membered
        "γ-thionolactone": "[SX1]=C1-O-[#6;!$(C=[O,S])]~[#6]~[#6]-1",
        "γ-thiolactam": "[SX1]=C1-N-[#6;!$(C=[O,S])]~[#6]~[#6]-1",
        "γ-dithiolactone": "[SX1]=C1-[SX2]-[#6;!$(C=[O,S])]~[#6]~[#6]-1",
        "thiofluorenone": "[SX1]=C1-[cX3H0]2:c:c:c:c:[cX3H0]:2-[cX3H0]3:c:c:c:c:[cX3H0]:3-1",
        "2-pyrroline-4-thione": "[SX1]=C1[CX4]N[#6;X3]~[#6;X3]1",
        "2-pyrroline-5-thione": "[SX1]=C1-N[#6;X3;!$(c1ccccc1)]~[#6;X3;!$(c1ccccc1)][CX4]1", # excludes indolethione
        "indole-2-thione": "[SX1]=C1-N-c2ccccc2-C1",
        "3-pyrrolinethione": "[SX1]=C1-N[CX4][#6;X3]~[#6;X3]1",
        "3-pyrazolethione": "[SX1]=c1:n:n:c:c:1",
        "4-imidazoline-2-thione": "[SX1]=c1nccn1",
        "4-oxazoline-2-thione": "[SX1]=c1ncco1",
        "4-thiazoline-2-thione": "[SX1]=c1nccs1",
        "1,3-dithiole-2-thione": "[SX1]=c1sccs1",
        "1,2,4-triazolidine-3-thione": "[SX1]=c1nn[c;!$(c=[O,S])]n1",
        "3-thioxo-1,2,4-triazolidinone": "[SX1]=c1nnc(=O)n1",
        "1,2,4-triazolidinedithione": "[SX1]=c1nnc(=[SX1])n1",
        "1,3,4-oxadiazole-2-thione": "[SX1]=c1nnco1",
        "1,3,4-thiadiazole-2-thione": "[SX1]=c1nncs1",
        "2-thiohydantoin": "O=C1-N-C(=[SX1])-[#7]~[#6;!$(C=O)]1",
        "4-thiohydantoin": "[SX1]=C1-N-C(=O)-[#7]~[#6;!$(C=O)]1",
        "dithiohydantoin": "[SX1]=C1-N-C(=[SX1])-[#7]~[#6;!$(C=O)]1",
        "rhodanine": "[SX1]=C1[SX2]CC(=O)N1",
    },
    "6&7-Membered": {
        # 6-membered
        "δ-thionolactone": "[SX1]=C1-O-[#6;!$(C=[O,S])]~[#6]~[#6]~[#6]-1",
        "δ-thiolactam": "[SX1]=C1-N-[#6;!$(C=[O,S])]~[#6]~[#6]~[#6]-1",
        "δ-dithiolactone": "[SX1]=C1-[SX2]-[#6;!$(C=[O,S])]~[#6]~[#6]~[#6]-1",
        "thio-1,2-benzoquinone": "[SX1]=C1-C(=O)-[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-1",
        "dithio-1,2-benzoquinone": "[SX1]=C1-C(=[SX1])-[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-1",
        "thio-1,4-benzoquinone": "[SX1]=C1-[#6;X3]=,:[#6;X3]-C(=O)-[#6;X3]=,:[#6;X3]-1",
        "dithio-1,4-benzoquinone": "[SX1]=C1-[#6;X3]=,:[#6;X3]-C(=[SX1])-[#6;X3]=,:[#6;X3]-1",
        "pyran-2-thione": "[SX1]=c1:c:c:c:c:o1",
        "pyran-4-thione": "[SX1]=c1:c:c:o:c:c1",
        "thiopyran-2-thione": "[SX1]=c1:c:c:c:c:s1",
        "thiopyran-4-thione": "[SX1]=c1:c:c:s:c:c1",
        "pyridine-2-thione": "[SX1]=c1:c:c:c:c:n1",
        "pyridine-4-thione": "[SX1]=c1:c:c:n:c:c1",
        "pyrimidine-2-thione": "[SX1]=c1:n:c:c:c:n1",
        "pyrimidine-4-thione": "[SX1]=c1:n:c:n:c:c1",
        "pyridazin-3-thione": "[SX1]=c1:n:n:c:c:c:1",
        "pyridazin-5-thione": "[SX1]=c1:c:n:n:c:c:1",
        "pyrazinethione": "[SX1]=c1:n:c:c:n:c:1",
        "2-thiobarbiturate": "[OX1,OH]~[#6;X3H0]1~N~[#6;X3H0](~[SX1,SH])~N~[#6;X3H0](~[OX1,OH])-C-1",
        "4-thiobarbiturate": "[SX1,SH]~[#6;X3H0]1~N~[#6;X3H0](~[OX1,OH])~N~[#6;X3H0](~[OX1,OH])-C-1",
        "2,4-dithiobarbiturate": "[SX1,SH]~[#6;X3H0]1~N~[#6;X3H0](~[SX1,SH])~N~[#6;X3H0](~[OX1,OH])-C-1",
        "4,6-dithiobarbiturate": "[SX1,SH]~[#6;X3H0]1~N~[#6;X3H0](~[OX1,OH])~N~[#6;X3H0](~[SX1,SH])-C-1",
        "trithiobarbiturate": "[SX1,SH]~[#6;X3H0]1~N~[#6;X3H0](~[SX1,SH])~N~[#6;X3H0](~[SX1,SH])-C-1",
        "thioammelide": "O~c1:n:c(~S):n:c(~N):n:1",
        "dithioammelide": "S~c1:n:c(~S):n:c(~N):n:1",
        "thioammeline": "S~c1:n:c(~N):n:c(~N):n:1",
        "monothio_cyanuric_acid": "S~c1:n:c(~O):n:c(~O):n:1",
        "dithio_cyanuric_acid": "S~c1:n:c(~S):n:c(~O):n:1",
        "trithio_cyanuric_acid": "S~c1:n:c(~S):n:c(~S):n:1",
        # 7-membered
        "tropothione": "[SX1]~c1:c:c:c:c:c:c:1",
    },
}

BIOMOLECULES: Dict[str, str] = {
    "Amino_acid": {
        # amino acids
        # N-terminus can be attached to anything, C-terminus cannot be aldehydes or ketones
        ## genetically encoded
        "glycine": "N-[CH2;!$(C1NC(=O)CNC(=O)1)]-C=[O;!$(O=C1NC(=O)NC1)]-[!$([#6,#1])]", # excludes hydantoin, diketopiperazine
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
        "lysine": "N-[CH](-[CH2][CH2][CH2][CH2][#7;!$(NC(=O)[CH]1[CH]([CH3])[CH2][CH]=N1)])-C(=O)-[!$([#6,#1])]", # excludes pyrrolysine
        "pyrrolysine": "N-[CH](-[CH2][CH2][CH2][CH2]NC(=O)[CH]1[CH]([CH3])[CH2][CH]=N1)-C(=O)-[!$([#6,#1])]",
        "arginine": "N-[CH](-[CH2][CH2][CH2]N~[CX3H0](~N)~N)-C(=O)-[!$([#6,#1])]",
        "proline":  "N1-[CH](-[CH2][CH2][CH2]1)-C(=O)-[!$([#6,#1])]",
        ## not genetically encoded
        "4-hydroxyproline": "N1-[CH](-[CH2][CH](-O)[CH2]1)-C(=O)-[!$([#6,#1])]",
        "5-hydroxylysine": "N-[CH](-[CH2][CH2][CH](-O)[CH2][#7])-C(=O)-[!$([#6,#1])]",
        "pyroglutamic_acid": "N1-[CH](-[CH2][CH2][C](=O)1)-C(=O)-[!$([#6,#1])]",
        "homoalanine": "N-[CH](-[CH2][CH3])-C(=O)-[!$([#6,#1])]",
        "norvaline": "N-[CH](-[CH2][CH2][CH3])-C(=O)-[!$([#6,#1])]",
        "homoserine": "N-[CH](-[CH2][CH2]O)-C(=O)-[!$([#6,#1])]",
        "homocysteine": "N-[CH](-[CH2][CH2]S)-C(=O)-[!$([#6,#1])]",
        "penicillamine": "N-[CH](-C([CH3])([CH3])S)-C(=O)-[!$([#6,#1])]",
        "selenomethionine": "N-[CH](-[CH2][CH2][Se][CH3])-C(=O)-[!$([#6,#1])]",
        "ornithine": "N-[CH](-[CH2][CH2][CH2][#7;!$(N-C(=[N,O])-N)])-C(=O)-[!$([#6,#1])]",
        "citrulline": "N-[CH](-[CH2][CH2][CH2]N-C(=O)-N)-C(=O)-[!$([#6,#1])]",
    },
    "Nucleobase": {
        # nucleobases
        "Purine": {
            ## purine bases
            "adenine": "N~[cX3H0]1:n:[c;!$(c~O)]:n:[cX3H0]2:n:c:n:[cX3H0]:1:2", # many tautomers, excludes isoguanine
            "guanine": "O~[cX3H0]1:n:[cX3H0](~N):n:[cX3H0]2:n:c:n:[cX3H0]:1:2", # many tautomers 
            "isoguanine": "N~[cX3H0]1:n:[cX3H0](~O):n:[cX3H0]2:n:c:n:[cX3H0]:1:2", # many tautomers
            "hypoxanthine": "O~[cX3H0]1:n:[c;!$(c~[O,N])]:n:[cX3H0]2:n:c:n:[cX3H0]:1:2", # many tautomers, excludes guanine and xanthine
            "xanthine": "O~[cX3H0]1:n:c(~O):n:[cX3H0]2:n:[c;!$(c~O)]:n:[cX3H0]:1:2", # many tautomers
            "uric_acid": "O~[cX3H0]1:n:c(~O):n:[cX3H0]2:n:c(~O):n:[cX3H0]:1:2", # many tautomers
        },
        "Pyrimidine": {
            ## pyrimidine bases
            "cytosine":    "[OX1,OH]~[cX3H0]1:n:[cX3H0](~N):[c;!$(c1ncnc1);!$(c1nccnc1)]:[c;!$(c1ncnc1);!$(c1nccnc1);!$(c~O)]:n:1", # many tautomers, excludes purines, isopterin
            "isocytosine": "[OX1,OH]~[cX3H0]1:n:[cX3H0](~N):n:[c;!$(c1ncnc1);!$(c1nccnc1);!$(c1NCCNc1)]:[c;!$(c1ncnc1);!$(c1nccnc1);!$(c1NCCNc1)]:1", # many tautomers, excludes purines, pterin
            "uracil": "[OX1,OH]~[cX3H0]1:n:[cX3H0;!$(c1nccnc1)](~[OX1,OH]):[c;!$(c-[CH3]);!$(c1ncnc1);!$(c1nccnc1);!$(c1NccNc1)]:[c;!$(c~[O,S]);!$(c1ncnc1)]:n:1", # many tautomers, excludes purines, pteridines
            "thymine": "[OX1,OH]~[cX3H0]1:n:[cX3H0](~[OX1,OH]):c(-[CH3]):c:n:1", # many tautomers 
        },
        "Thio": {
            ## thio analogues
            "thioguanine": "[SX1,SH]~[cX3H0]1:n:[cX3H0](~N):n:[cX3H0]2:n:c:n:[cX3H0]:1:2", # many tautomers 
            "thiocytosine":    "[SX1,SH]~[cX3H0]1:n:[cX3H0](~N):[c;!$(c1ncnc1);!$(c1nccnc1)]:[c;!$(c1ncnc1);!$(c1nccnc1);!$(c~O)]:n:1", # many tautomers, excludes purines, isopterin
            "2-thiouracil": "[SX1,SH]~[cX3H0]1:n:[cX3H0;!$(c1nccnc1)](~[OX1,OH]):[c;!$(c1ncnc1);!$(c1nccnc1);!$(c1NccNc1)]:[c;!$(c~[O,S]);!$(c1ncnc1)]:n:1", # many tautomers, excludes purines, pteridines
            "4-thiouracil": "[OX1,OH]~[cX3H0]1:n:[cX3H0;!$(c1nccnc1)](~[SX1,SH]):[c;!$(c1ncnc1);!$(c1nccnc1);!$(c1NccNc1)]:[c;!$(c~[O,S]);!$(c1ncnc1)]:n:1", # many tautomers, excludes purines, pteridines
            "dithiouracil": "[SX1,SH]~[cX3H0]1:n:[cX3H0;!$(c1nccnc1)](~[SX1,SH]):[c;!$(c1ncnc1);!$(c1nccnc1);!$(c1NccNc1)]:[c;!$(c~[O,S]);!$(c1ncnc1)]:n:1", # many tautomers, excludes purines, pteridines
        },

    },
    "Sugar": {
        "C3": {
            ## triose
            "glyceraldehyde": "O-[CH2]-[CH](-O)-[CH]=[O,N]",
            "glycerol": "O-[CH2]-[CH](-O)-[CH2]-O",
            "glyceric_acid": "O-[CH2]-[CH](-O)-C(=O)-[!#1;!#6]",
        },
        "C4": {
            ## tetrose
            "aldotetrose": "[$([CH]=[O,N]),$([CH](-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH2](O)", #including hemiacetal, acetal, etc. and cyclic forms
            "ketotetrose": "O[CH2][$(C=[O,N]),$(C(-[O,S,N,n])-[O,S,N,n])][CH](O)[CH2](O)", #including hemiketal, ketal, etc, and cyclic forms
            "tetritol": "O-[CH2]-[CH](-O)-[CH](-O)-[CH2]-O",
            "tetro_aldonic_acid": "[!#1;!#6]-C(=O)-[CH](O)[CH](O)[CH2](O)",
            "tetro_uronic_acid": "[$([CH]=[O,N]),$([CH](-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)-C(=O)-[!#1;!#6]",
            "tartaric_acid": "[!#1;!#6]-C(=O)-[CH](O)[CH](O)-C(=O)-[!#1;!#6]",
        },
        "C5": {
            ## pentose
            "aldopentose": "[$([CH]=[O,N]),$([CH](-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH](O)[CH2](O)", #including hemiacetal, acetal, etc. and cyclic forms
            "ketopentose": "O[CH2][$(C=[O,N]),$(C(-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH2](O)", #including hemiketal, ketal, etc, and cyclic forms
            "pentitol": "O-[CH2]-[CH](-O)-[CH](-O)-[CH](-O)-[CH2]-O",
            "pento_aldonic_acid": "[!#1;!#6]-C(=O)-[CH](O)[CH](O)[CH](O)[CH2](O)",
            "penturonic_acid": "[$([CH]=O),$([CH](-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH](O)-C(=O)-[!#1;!#6]",
            "pentulosonic_acid": "[!#1;!#6]-C(=O)-[$(C=[O,N]),$(C(-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH2](O)",
            "pentaric_acid": "[!#1;!#6]-C(=O)-[CH](O)[CH](O)[CH](O)-C(=O)-[!#1;!#6]",
        },
        "C6": {
            ## hexose
            "aldohexose": "[$([CH]=[O,N]),$([CH](-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH](O)[CH](O)[CH2](O)", #including hemiacetal, acetal, etc. and cyclic forms
            "ketohexose": "O[CH2][$(C=[O,N]),$(C(-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH](O)[CH2](O)", #including hemiketal, ketal, etc. and cyclic forms
            "hexitol": "O-[CH2]-[CH](-O)-[CH](-O)-[CH](-O)-[CH](-O)-[CH2]-O",
            "inositol": "O-[CH]1-[CH](-O)-[CH](-O)-[CH](-O)-[CH](-O)-[CH]1(-O)",
            "hexo_aldonic_acid": "[!#1;!#6]-C(=O)-[CH](O)[CH](O)[CH](O)[CH](O)[CH2](O)",
            "hexuronic_acid": "[$([CH]=O),$([CH](-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH](O)[CH](O)-C(=O)-[!#1;!#6]",
            "hexulosonic_acid": "[!#1;!#6]-C(=O)-[$(C=[N,O]),$(C(-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH](O)[CH2](O)",
            "hexaric_acid": "[!#1;!#6]-C(=O)-[CH](O)[CH](O)[CH](O)[CH](O)-C(=O)-[!#1;!#6]",
        },
        "C7": {
            ## heptose
            "aldoheptose": "[$([CH]=[O,N]),$([CH](-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH](O)[CH](O)[CH](O)[CH2](O)", #including hemiacetal, acetal, etc. and cyclic forms
            "ketoheptose": "O[CH2][$(C=[O,N]),$(C(-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH](O)[CH](O)[CH2](O)", #including hemiketal, ketal, etc. and cyclic forms
            "heptitol": "O-[CH2]-[CH](-O)-[CH](-O)-[CH](-O)-[CH](-O)-[CH](-O)-[CH2]-O",
            "hepto_aldonic_acid": "[!#1;!#6]-C(=O)-[CH](O)[CH](O)[CH](O)[CH](O)[CH](O)[CH2](O)",
            "hepturonic_acid": "[$([CH]=O),$([CH](-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH](O)[CH](O)[CH](O)-C(=O)-[!#1;!#6]",
            "heptulosonic_acid": "[!#1;!#6]-C(=O)-[$(C=[O,N]),$(C(-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH](O)[CH](O)[CH2](O)",
            "heptaric_acid": "[!#1;!#6]-C(=O)-[CH](O)[CH](O)[CH](O)[CH](O)[CH](O)-C(=O)-[!#1;!#6]",
        },
        "Deoxy/Amino": {
            ## deoxysugars / aminosugars 
            "2-deoxy-aldopentose": "[$([CH]=O),$([CH](-[O,S,N,n])-[O,S,N,n])][CH2][CH](O)[CH](O)[CH2]O",
            "6-deoxy-aldohexose":   "[$([CH]=O),$([CH](-[O,S,N,n])-[O,S,N,n])][CH](O)[CH](O)[CH](O)[CH](O)[CH3]",
            "hexosamine": "[$([CH]=O),$([CH](-[O,S,N,n])-[O,S,N,n])][CH](N)[CH](O)[CH](O)[CH](O)[CH2](O)",
            "6-deoxy-hexosamine": "[$([CH]=O),$([CH](-[O,S,N,n])-[O,S,N,n])][CH](N)[CH](O)[CH](O)[CH](O)[CH3]",
            "3-deoxy-octulosonic_acid": "[!#1;!#6]-C(=O)-[$(C=[O,N]),$(C(-[O,S,N,n])-[O,S,N,n])][CH2][CH](O)[CH](O)[CH](O)[CH](O)[CH2](O)",
            "neuraminic_acid": "[!#1;!#6]-C(=O)-[$(C=[O,N]),$(C(-[O,S,N,n])-[O,S,N,n])][CH2][CH](O)[CH](N)[CH](O)[CH](O)[CH](O)[CH2](O)",
        },
    },

    "Fat": {
        "Lipid": {
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
        },
        "Fatty_acid": {
            "Saturated": {
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
            },
            "Unsaturated": {
                    ### unsaturated (all cis)
                "palmitoleoyl": r"[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]\[CH]=[CH]/[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 16:1 omega-7
                "sapienoyl": r"[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]\[CH]=[CH]/[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 16:1 omega-10
                "oleoyl": r"[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]\[CH]=[CH]/[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 18:1 omega-9
                "linoleoyl": r"[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 18:2 omega-6
                "alpha-linoleoyl": r"[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]-[CH3]", # 18:3 omega-3
                "gamma-linoleoyl": r"[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 18:3 omega-6
                "stearidonoyl": r"[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]-[CH]=[CH]-[CH2]-[CH2]=[CH2]-[CH2]-[CH3]", # 18:4  omega-3
                "dihomo-gamma-linoleoyl": r"[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 20:3  omega-6
                "arachidonoyl": r"[*]-C(=O)-[CH2]-[CH2]-[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 20:4  omega-6
                "eicosapentaenoyl": r"[*]-C(=O)-[CH2]-[CH2]-[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]-[CH3]", # 20:5  omega-3
                "adrenoyl": r"[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # 22:4  omega-6
                "docosapentaenoyl": r"[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]-[CH3]", # 22:5  omega-3
                "docosahexaenoyl": r"[*]-C(=O)-[CH2]-[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]\[CH]=[CH]/[CH2]-[CH3]", # 22:6  omega-3
                ### trans fatty acids
                "trans-palmitoleoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]/[CH]=[CH]/[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # trans 16:1 omega-7
                "elaidoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]/[CH]=[CH]/[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # trans 18:1 omega-9
                "vaccenoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]/[CH]=[CH]/[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # trans 18:1 omega-7
                "rumenoyl": r"[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]\[CH]=[CH]/[CH]=[CH]/-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # trans11 18:2 omega-7
                "linoleladoyl": "[*]-C(=O)-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]/[CH]=[CH]/[CH2]/[CH]=[CH]/[CH2]-[CH2]-[CH2]-[CH2]-[CH3]", # trans,trans 18:2 omega-6
            }
        },
    },

    "Flavonoid": {
        ## flavonoids
        "flavan_core": "[#8]1~c2ccccc2~[#6]~[#6]~[#6]~1~c2ccccc2",
        "isoflavan_core": "[#8]1~c2ccccc2~[#6]~[#6](~c2ccccc2)~[#6]~1",
        "neoflavan_core": "[#8]1~c2ccccc2~[#6](~c2ccccc2)~[#6]~[#6]~1",
        "chalcone_core": "c1ccccc1-C(=O)-[C;!$([#6]1~[*]~c2ccccc2~[#6](=O)1)]=[C;!$([#6]1~[*]~c2cccccc2~[#6](=O)~[#6]=1)]-c1ccccc1", # excludes flavan and aurone
        "aurone_core": "O1-c2ccccc2-C(=O)-C1=C-c2ccccc2", 
    },

    "Terpenoid": {
        "Branch": {
            ## isoprenoid chains
            "prenyl":   "[CH3]-C(-[CH3])=[CH]-[CH2;!$([CH2]-[CH2]-C(-[CH3])=[CH]-C)]", # excludes geranyl
            "isopentenyl": "[CH2]=C(-[CH3])-[CH2]-[CH2]-[!C,$([C;H0,H1]);!#1]",
            "geranyl":  "[CH3]-C(-[CH3])=[CH]-[CH2]-[CH2]-C(-[CH3])=[CH]-[C;!$([CH2]-[CH2]-C(-[CH3])=[CH]-C)]", # excludes farnesyl
            "myrcanyl": "[CH3]-[CH](-[CH3])-[CH2]-[CH2]-[CH2]-[CH](-[CH3])-[CH2]-[C;!$([CH2]-[CH2]-[CH](-[CH3])-[CH2]-C)]", # excludes farnesanyl
            "linalyl": "[CH3]-C(-[CH3])=[CH]-[CH2]-[CH2]-[CH0](-[*])(-[CH3])-[CH]=[CH2]",
            "farnesyl": "[CH3]-C(-[CH3])=[CH]-[CH2]-[CH2]-C(-[CH3])=[CH]-[CH2]-[CH2]-C(-[CH3])=[CH]-[C;!$([CH2]-[CH2]-C(-[CH3])=[CH]-C)]", # excludes geranylgeranyl
            "farnesanyl": "[CH3]-[CH](-[CH3])-[CH2]-[CH2]-[CH2]-[CH](-[CH3])-[CH2]-[CH2]-[CH2]-[CH](-[CH3])-[CH2]-[CH2]-[C;!$([CH2]-[CH2]-[CH](-[CH3])-[CH2]-C);!$([CH2]-[CH2]-C(-[CH3])=[CH]-[CH2])]", # excludes phytyl, phytanyl
            "nerolidyl": "[CH3]-C(-[CH3])=[CH]-[CH2]-[CH2]-C(-[CH3])=[CH]-[CH2]-[CH2]-[CH0](-[*])(-[CH3])-[CH]=[CH2]",
            "geranylgeranyl": "[CH3]-C(-[CH3])=[CH]-[CH2]-[CH2]-C(-[CH3])=[CH]-[CH2]-[CH2]-C(-[CH3])=[CH]-[CH2]-[CH2]-C(-[CH3])=[CH]-C",
            "phytyl": "[CH3]-[CH](-[CH3])-[CH2]-[CH2]-[CH2]-[CH](-[CH3])-[CH2]-[CH2]-[CH2]-[CH](-[CH3])-[CH2]-[CH2]-[CH2]-C(-[CH3])=[CH]-[CH2]-[!#1]",
            "phytanyl": "[CH3]-[CH](-[CH3])-[CH2]-[CH2]-[CH2]-[CH](-[CH3])-[CH2]-[CH2]-[CH2]-[CH](-[CH3])-[CH2]-[CH2]-[CH2]-[CH](-[CH3])-[CH2]-[CH2]-[!#1]",
        },
        "Scaffold": {
            "Monoterpene": {
                ### mono- or acyclic
                "myrcane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])~[#6])]",
                "p-menthane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1", 
                ### bicyclic
                "bornane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6]12~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1)~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~2",   
                "pinane_terpenoid": "[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]12~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1)~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~2",   
                "carane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "iridane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "secoiridane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])~[#6])]",
                "thujane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]~2(~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "fenchane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6]12~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1)~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~2",
            },
            "Sesquiterpene": {
                ### mono- or acyclic
                "farnesane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])~[#6])]",
                "germacrane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "bisabolane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1", 
                "humulane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "elemane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]1(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                ### bicyclic
                "eudesmane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]2(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "guaiane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "pseudoguaiane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "drimane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6]2(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "caryophyllane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "cadinane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~1~2",
                "eremophilane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])~1~2",
                "daucane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "oplopane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~1~2",
                "acorane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]2(~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])~[#6])])~2)~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "tremulane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~1~2",
                ### tricyclic
                "aristolane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6](~[#6;!$([#6](~[#6])~[#6])])~1~2",
                "lindenane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]2(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "cedrane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]23~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~3",
                "aromadendrane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~1",
                "chamigrane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]2(~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~2)~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "illudane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6]3(~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~3)~1",
                "protoilludane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6]3(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~1",
                "hirsutane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6](~[#6;!$([#6](~[#6])~[#6])])~1~2",
                "patchoulane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~2)~3~1",
            },
            "Diterpene": {
                ### mono- or acyclic
                "phytane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])~[#6])]",
                "cembrane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "retinoid_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~1",
                ### bicyclic
                "labdane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])~2~1",
                "clerodane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]1(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]2(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~1",
                "halimane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])1~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~1",
                "dolabellane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]2(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "casbane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "jatrophane_terpenoid": "[#6;!$([#6](~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~1~2",
                "eunicellane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~1",
                "briarane": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]2(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~1~2",
                ### tricyclic
                "abietane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6]3(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "pimarane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]1(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6]3(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "taxane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~2)~1",
                "cassane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6]3(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~1",
                "daphnane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "podocarpane_terpenoid": "[#6;!$([#6](~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]2(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~1",
                "rosane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]1(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]2(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "fusicoccane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])][#6])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]2(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~1~2",
                "lathyrane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "cyathane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]2(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]3(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~1~2",
                ### tetracyclic
                "kaurane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6]4(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~4~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])(~[#6])~[#6])]~2)~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "beyerane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6]12~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6]4(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~4~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])(~[#6])~[#6])]~2)~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "atisane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6]4(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~4~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~2)~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "stemodane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]23~[#6]4(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~4~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~2",
                "gibberellin_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[$([#6]~[#6;!$([#6](~[#6])~[#6])]),$([#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])])]4~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~4~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6](~[#6;!$([#6](~[#6])(~[#6])~[#6])]~2)~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "ingenane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6]4(~[#6;!$([#6](~[#6])(~[#6])~[#6])]~2)~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~4~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "grayanane_terpenoid": "[#6;!$([#6](~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]4~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]~3(~[#6;!$([#6](~[#6])(~[#6])~[#6])]~4)~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~1",
                ### pentacyclic
                "trachylobane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6]15~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6]4(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~4~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~5)~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
            },
            "Sesterterpene": {
                ## sesterterpenoids (C25)
                "scalarane_terpenoid": "[#6;!$([#6](~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]3(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]4~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6](~[#6;!$([#6](~[#6])~[#6])])~4~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6](~[#6;!$([#6](~[#6])~[#6])])~1~2",
                "ophiobolane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]2(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~1~2",
                "cheilanthane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6]3(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])~2~1",
            },
            "Triterpene": {
                ### acyclic
                "squalane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])~[#6])]",
                ### tetracyclic
                "lanostane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6]2(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6]4(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~4~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6](~[#6;!$([#6](~[#6])~[#6])])~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "dammarane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6]4(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~4~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])~3~[#6](~[#6;!$([#6](~[#6])~[#6])])~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "cucurbitane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6]2(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]3(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]4~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~4~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6](~[#6;!$([#6](~[#6])~[#6])])~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                ### pentacyclic
                "cycloartane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6]2(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]3(~[#6;!$([#6](~[#6])(~[#6])~[#6])]5)~[#6]4~5~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~4~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6](~[#6;!$([#6](~[#6])~[#6])])~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "oleanane_terpenoid": "[#6;!$([#6](~[#6])(~[#6])~[#6])]1~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]4~[#6]5(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~5~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])~4~[#6](~[#6;!$([#6](~[#6])~[#6])])~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "ursane_terpenoid": "[#6;!$([#6](~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]4~[#6]5(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~5~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])~4~[#6](~[#6;!$([#6](~[#6])~[#6])])~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "lupane_terpenoid": "[#6;!$([#6](~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]4~[#6]5(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~5~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])~4~[#6](~[#6;!$([#6](~[#6])~[#6])])~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])~2~[#6;!$([#6](~[#6])(~[#6])~[#6])]~1",
                "friedelane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6]3(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6]4(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]5~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])~5~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])~4~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])~2~1",
                "hopane_terpenoid": "[#6;!$([#6](~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])2~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]4~[#6]5(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~5~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])~4~[#6](~[#6;!$([#6](~[#6])~[#6])])~3~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~2~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])~[#6])])~1",
            },
            "Tetraterpene": {
            "lycopane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])~[#6])]",
            "carotane_terpenoid": "[#6;!$([#6](~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]~1~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])]2~[#6;!$([#6](~[#6])(~[#6])(~[#6])~[#6])](~[#6;!$([#6](~[#6])~[#6])])~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6;!$([#6](~[#6])(~[#6])~[#6])]~[#6](~[#6;!$([#6](~[#6])~[#6])])(~[#6;!$([#6](~[#6])~[#6])])~2",
            }
        }

    },

    "Alkaloid/Privileged": {
        "phenethylamine": "[N;!$(N1ccCC1);!$(N1CccCC1);!$(N1~C~CC2ccCC(C2)1);!$(N1~C~C~C~C2~C1~C-c3cnc4cccc-2c34)]-[C;!$(C=O);!$([CH](N)(C(=O)-[!#6;!#1])-[CH2]-c1ccccc1)]-C-c1ccccc1", # excludes phenylalanine, tyrosine, indoline, tetrahydroisoquinoline, 6,7-benzomorphan, ergoline
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
        "granatane": "[CH3]-N1-C2-[#6]-,=,:[#6]-,=,:[#6]-C(-[#6]-,=,:[#6]-,=,:[#6]-2)-1", 
        "morphan": "C12-[#7]-,=,:[#6;!$(c1ccccc1)]-,=,:[#6;!$(c1ccccc1)]-C(-C2)-[#6;!$(c1ccccc1)]-,=,:[#6;!$(c1ccccc1)]-,=,:[#6]-1",
        "3,4-benzomorphan": "C12-[#7]-,:c3ccccc3-C(-C2)-[#6]-,=,:[#6]-,=,:[#6]-1",
        "6,7-benzomorphan": "C12-[#7]-,=,:[#6]-,=,:[#6]-[C;!$(C123-[#6]~[#6]~[#6]~[#6]-C-2-C(-[#7]~[#6]~[#6]-3)-C-cc-1)](-C2)-c3ccccc3-,:[#6]-1", # excludes morphinan
        "morphinan": "C12-[#7]-,=,:[#6]-,=,:[#6]-C(-C3-2)(-[#6]-,=,:[#6]-,=,:[#6]-,=,:[#6]-3)-c4ccccc4-,:[#6]-1",
        "aporphine": "N1-C-C-c2cccc(c23)~c4ccccc4~[#6]~[#6]~3-1",
        "ergoline": "n1cc(-C-,=C3-,=N-,=,:[#6]-,=,:[#6]-,=,:[#6]-,=C4-,=3)c2c-4cccc12",
        "ibogalog": "[#7]1-,=,:[#6]-,=,:[#6]-c2c3ccccc3nc2-[#6]-,=,:[#6]-,=,:1",
        "protoberberine": "c1ccccc-c2cc3ccccc3c[n+]2-C-C-1",
        "benzodiazepine": "[#7]1-c2ccccc2-[#6]~[#7]~[#6]~[#6]~1",
        "thienodiazepine": "[#7]1-c2sccc2-[#6]~[#7]~[#6]~[#6]~1",
        "benzothiazepine": "[#7]1-c2ccccc2-S-[#6]~[#6]~[#6]~1",
        "4-aza-steroid": "[#6]1~[#6]~[#6]~[#7]~[#6]2~[#6]~[#6]~[#6]3~[#6]4~[#6]~[#6]~[#6]~[#6]~4~[#6]~[#6]~[#6]~3~[#6]~2~1",
        
    },

    "Misc": {
        # misc
        "choline": "O-[CH2]-[CH2]-[N+](-[CH3])(-[CH3])-[CH3]",
        "taurine": "O-[$([SX4](=O)=O),$([SX4+](=O)-[O-]),$([SX4+2](-[O-])-[O-])]-[CH2]-[CH2]-N",
        "biotinyl": "O=C1-N-[CH]2-[CH](-[CH2]-[CH2]-[CH2]-[CH2]-C=O)-S-[CH2]-[CH]2-N1",
        "GABA": "N-[CH2;!r5]-[CH2;!r5]-[CH2;!r5]-[C;!$(C1(=O)CCCN1)](=O)-[!#1;!#6]",
        "carnithine_core": "[!#1;!#6]-C(=O)-[CH2]-[CH](-O)-[CH2]-[N+](-[CH3])(-[CH3])-[CH3]",
        "pantothenic_core": "O-[CH2]-C(-[CH3])(-[CH3])-[CH](-O)-C(=O)-N-[CH2]-[CH2]-C(=O)-[!#1;!#6]",
        "mevalonic_core": "O-[CH2]-[CH2]-C(-O)(-[CH3])-[CH2]-C=O-[!#1;!#6]",
        "ascorbic_core": "O1-[$(C=[O,N]),$(C(-[O,S,N,n])-[O,S,N,n])]-[CX3H0](~O)~[CX3H0](~O)-[CH]1-[CH](-O)-[CH2]-O",
        "citric_core": "[!#1;!#6]-C(=O)-[CH2]-C(-O)(-C(=O)-[!#1;!#6])-[CH2]-C(=O)-[!#1;!#6]",
        "homocitric_core": "[!#1;!#6]-C(=O)-[CH2]-[CH2]-C(-O)(-C(=O)-[!#1;!#6])-[CH2]-C(=O)-[!#1;!#6]",
    }
}

ALL = {
    "CORE": CORE,
    "BRANCHES" : BRANCHES,
    "MAIN_GROUP": MAIN_GROUP,
    "HOMOAROMATICS": HOMOAROMATICS,
    "HETEROAROMATICS": HETEROAROMATICS,
    "ALIPHATIC_RINGS": ALIPHATIC_RINGS,
    "HETEROALIPHATIC_RINGS": HETEROALIPHATIC_RINGS,
    "OXO_RINGS": OXO_RINGS,
    "THIOXO_RINGS": THIOXO_RINGS,
    "BIOMOLECULES": BIOMOLECULES,
}

from rdkit import Chem
def get_counts(smiles):
    
    def _get_counts(mol, smarts_dict):
        counts = dict()
        first_entry = next(iter(smarts_dict.values()))
        if isinstance(first_entry, str):
            for name, smart in smarts_dict.items():
                sub = Chem.MolFromSmarts(smart)
                if (count := len(mol.GetSubstructMatches(sub, useChirality=True))) > 0:
                    counts[name] = count
        else:
            for name, subdict in smarts_dict.items():
                if (count_dict := _get_counts(mol, subdict)):
                    counts[name] = count_dict
        return counts

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    return _get_counts(mol, ALL)
