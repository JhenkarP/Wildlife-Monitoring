# Felidae classification flowchart

The classifier follows the taxonomic path below. The final label always contains both the common name and the scientific name.

```mermaid
flowchart TD
    A[Input image] --> B[Animal detector]
    B --> C{Cat / non-cat?}
    C -->|Non-cat| D[Reject: not a cat]
    C -->|Cat| E[Family: Felidae]
    E --> F{Subfamily}
    F -->|Pantherinae| G[Panthera lineage]
    F -->|Felinae| H[Seven Felinae lineages]
    G --> I[Neofelis or Panthera genus]
    H --> J[Caracal, Ocelot, Bay cat, Lynx, Puma, Leopard cat, or Domestic cat lineage]
    I --> K[Species label: common name + scientific name]
    J --> K
    K --> L{Confidence sufficient?}
    L -->|Yes| M[Return species prediction]
    L -->|No| N[Unknown / insufficient evidence]

    subgraph Pantherinae species
      P1[Neofelis diardi - Sunda clouded leopard]
      P2[Neofelis nebulosa - Clouded leopard]
      P3[Panthera tigris - Tiger]
      P4[Panthera uncia - Snow leopard]
      P5[Panthera pardus - Leopard]
      P6[Panthera leo - Lion]
      P7[Panthera onca - Jaguar]
    end

    subgraph Felinae representative path
      F1[Leopardus pardalis - Ocelot]
      F2[Lynx lynx - Eurasian lynx]
      F3[Acinonyx jubatus - Cheetah]
      F4[Prionailurus bengalensis - Leopard cat]
      F5[Felis catus - Domestic cat]
    end
```

The complete species list, including every genus and lineage, is in `felidae_species.json` and is rendered by `app.py`.
