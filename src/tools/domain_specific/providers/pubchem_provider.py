"""
PubChem knowledge provider for chemistry domain information.
"""

import os
import json
import logging
import httpx
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from ..base_provider import DomainKnowledgeProvider

logger = logging.getLogger("co_scientist")

class PubChemProvider(DomainKnowledgeProvider):
    """
    PubChem knowledge provider for chemical compounds, substances, and bioassays.
    
    This provider interfaces with the PubChem PUG REST API to search and retrieve
    data about chemical compounds, substances, bioassays, and related information.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the PubChem provider.
        
        Args:
            config (Dict[str, Any], optional): Configuration for the provider. Defaults to None.
        """
        super().__init__(domain="chemistry", config=config)
        
        # PubChem API base URL
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        
        # HTTP client for API requests
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Entity types in PubChem
        self.entity_types = {
            "compound": "chemical compound",
            "substance": "chemical substance",
            "assay": "bioassay",
            "gene": "gene",
            "protein": "protein",
            "pathway": "pathway",
            "taxonomy": "taxonomy"
        }
        
        # Citation format templates
        self.citation_templates = {
            "apa": "PubChem ({year}). {title} ({identifier}). {source}. Retrieved from {url}",
            "mla": "PubChem. \"{title}.\" {source}, {year}, {identifier}. {url}. Accessed {access_date}.",
            "chicago": "PubChem. \"{title}.\" {source}, {year}. {url}.",
            "vancouver": "PubChem. {title}. {source}. {year}. {identifier}. Available from: {url}"
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the PubChem provider.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Test connection to PubChem API with a simple query
            url = f"{self.base_url}/compound/name/aspirin/cids/JSON"
            response = await self.client.get(url)
            response.raise_for_status()
            
            # Parse response to confirm it's valid
            data = response.json()
            if "IdentifierList" in data and "CID" in data["IdentifierList"]:
                self._is_initialized = True
                logger.info("PubChem provider initialized successfully")
                return True
            else:
                logger.error("PubChem API response did not contain expected structure")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize PubChem provider: {str(e)}")
            return False
    
    async def search(self, query: str, limit: int = 10, entity_type: str = "compound") -> List[Dict[str, Any]]:
        """
        Search PubChem for chemicals or other entities.
        
        Args:
            query (str): The search query
            limit (int, optional): Maximum number of results. Defaults to 10.
            entity_type (str, optional): Type of entity to search for 
                                        (compound, substance, assay). Defaults to "compound".
            
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        if not self._is_initialized:
            await self.initialize()
            
        # Validate entity type
        if entity_type not in self.entity_types:
            entity_type = "compound"
            
        # Clean and validate query
        if not query or len(query.strip()) < 2:
            logger.warning("Empty or too short query provided to PubChem provider")
            return []
        
        # Clean the query - PubChem is sensitive to certain characters
        clean_query = query.replace("'", "").replace('"', "").replace('[', "").replace(']', "").strip()
        
        # Check if the query is a research question or concept rather than a chemical name
        research_keywords = ["disease", "research", "study", "therapy", "treatment", "mechanism", 
                           "pathway", "diagnosis", "method", "approach", "analysis", "machine learning",
                           "artificial intelligence", "cognitive", "neural", "protein", "gene",
                           "dysfunction", "discovery", "development", "identification"]
                           
        # If the query looks like a research question not a chemical, return empty results
        for keyword in research_keywords:
            if keyword.lower() in clean_query.lower():
                logger.info(f"Query '{query}' appears to be a research concept, not a chemical entity")
                return []
                
        try:
            # Get IDs for the query
            search_url = f"{self.base_url}/{entity_type}/name/{clean_query}/cids/JSON"
            
            # For assays, we use a different endpoint
            if entity_type == "assay":
                search_url = f"{self.base_url}/assay/description/{clean_query}/aids/JSON"
            
            try:
                response = await self.client.get(search_url)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    # A 404 likely means the query doesn't match any chemical compounds
                    logger.info(f"No chemical entities found in PubChem for '{query}'")
                    return []
                else:
                    # Re-raise for other status codes
                    raise
            
            # Extract IDs based on entity type
            ids = []
            if entity_type == "compound":
                ids = data.get("IdentifierList", {}).get("CID", [])
            elif entity_type == "substance":
                ids = data.get("IdentifierList", {}).get("SID", [])
            elif entity_type == "assay":
                ids = data.get("IdentifierList", {}).get("AID", [])
            
            # Limit the number of IDs
            ids = ids[:limit]
            
            if not ids:
                return []
                
            # Get detailed information for each ID
            results = []
            for entity_id in ids:
                entity = await self.get_entity(str(entity_id), entity_type)
                if entity:
                    results.append(entity)
                    
            return results
            
        except Exception as e:
            logger.error(f"Error searching PubChem: {str(e)}")
            return []
    
    async def get_entity(self, entity_id: str, entity_type: str = "compound") -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed information about a specific PubChem entity.
        
        Args:
            entity_id (str): The ID of the entity (CID for compounds, SID for substances, etc.)
            entity_type (str, optional): Type of entity (compound, substance, assay). Defaults to "compound".
            
        Returns:
            Optional[Dict[str, Any]]: Entity information or None if not found
        """
        if not self._is_initialized:
            await self.initialize()
            
        # Validate entity type
        if entity_type not in self.entity_types:
            entity_type = "compound"
            
        try:
            # Build URL based on entity type
            if entity_type == "compound":
                url = f"{self.base_url}/compound/cid/{entity_id}/record/JSON"
            elif entity_type == "substance":
                url = f"{self.base_url}/substance/sid/{entity_id}/record/JSON"
            elif entity_type == "assay":
                url = f"{self.base_url}/assay/aid/{entity_id}/description/JSON"
            else:
                logger.error(f"Unsupported entity type: {entity_type}")
                return None
                
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Process data based on entity type
            result = {
                "id": entity_id,
                "entity_type": entity_type,
                "source": "pubchem",
                "provider": "pubchem",
                "url": f"https://pubchem.ncbi.nlm.nih.gov/{entity_type}/{entity_id}"
            }
            
            if entity_type == "compound":
                # Extract compound information
                if "PC_Compounds" in data and data["PC_Compounds"]:
                    compound = data["PC_Compounds"][0]
                    
                    # Get compound identifiers
                    for prop in compound.get("props", []):
                        if prop.get("urn", {}).get("label") == "IUPAC Name":
                            result["name"] = prop.get("value", {}).get("sval", "")
                        elif prop.get("urn", {}).get("label") == "InChI":
                            result["inchi"] = prop.get("value", {}).get("sval", "")
                        elif prop.get("urn", {}).get("label") == "InChIKey":
                            result["inchikey"] = prop.get("value", {}).get("sval", "")
                        elif prop.get("urn", {}).get("label") == "SMILES":
                            result["smiles"] = prop.get("value", {}).get("sval", "")
                        elif prop.get("urn", {}).get("label") == "Molecular Formula":
                            result["formula"] = prop.get("value", {}).get("sval", "")
                        elif prop.get("urn", {}).get("label") == "Molecular Weight":
                            result["molecular_weight"] = prop.get("value", {}).get("fval", "")
                    
                    # Set title (prefer name, fall back to ID)
                    result["title"] = result.get("name", f"Compound {entity_id}")
                    
                    # Get synonyms
                    synonym_url = f"{self.base_url}/compound/cid/{entity_id}/synonyms/JSON"
                    try:
                        syn_response = await self.client.get(synonym_url)
                        syn_response.raise_for_status()
                        syn_data = syn_response.json()
                        
                        if "InformationList" in syn_data and "Information" in syn_data["InformationList"]:
                            synonyms = syn_data["InformationList"]["Information"][0].get("Synonym", [])
                            result["synonyms"] = synonyms[:20]  # Limit to 20 synonyms
                    except Exception as e:
                        logger.warning(f"Failed to get synonyms for compound {entity_id}: {str(e)}")
                    
                    # Fetch some basic properties
                    props_url = f"{self.base_url}/compound/cid/{entity_id}/property/MolecularFormula,MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,TPSA/JSON"
                    try:
                        props_response = await self.client.get(props_url)
                        props_response.raise_for_status()
                        props_data = props_response.json()
                        
                        if "PropertyTable" in props_data and "Properties" in props_data["PropertyTable"]:
                            properties = props_data["PropertyTable"]["Properties"][0]
                            result["properties"] = properties
                    except Exception as e:
                        logger.warning(f"Failed to get properties for compound {entity_id}: {str(e)}")
                    
            elif entity_type == "substance":
                # Extract substance information
                if "PC_Substances" in data and data["PC_Substances"]:
                    substance = data["PC_Substances"][0]
                    
                    # Get substance name
                    if "name" in substance:
                        result["name"] = substance["name"]
                    
                    # Get source information
                    if "source" in substance:
                        source_name = substance["source"].get("db", {}).get("name", "")
                        source_id = substance["source"].get("db", {}).get("source_id", {}).get("str", "")
                        result["source_db"] = source_name
                        result["source_id"] = source_id
                    
                    # Set title (prefer name, fall back to ID)
                    result["title"] = result.get("name", f"Substance {entity_id}")
                    
                    # Get compounds associated with this substance
                    if "compound" in substance:
                        result["compound_ids"] = [cid["id"]["id"]["cid"] for cid in substance["compound"]]
                    
            elif entity_type == "assay":
                # Extract assay information
                if "PC_AssayContainer" in data and data["PC_AssayContainer"]:
                    assay = data["PC_AssayContainer"][0].get("assay", {})
                    desc = assay.get("descr", {})
                    
                    # Get assay name and description
                    result["name"] = desc.get("name", "")
                    result["title"] = result["name"]
                    result["description"] = desc.get("description", "")
                    result["protocol"] = desc.get("protocol", "")
                    result["comments"] = desc.get("comment", "")
                    
                    # Get assay target information
                    if "target" in desc:
                        targets = []
                        for target in desc["target"]:
                            target_dict = {
                                "name": target.get("name", ""),
                                "type": target.get("mol_id", {}).get("descr", "")
                            }
                            targets.append(target_dict)
                        result["targets"] = targets
                    
                    # Get citations
                    if "xref" in desc:
                        citations = []
                        for xref in desc["xref"]:
                            if xref.get("xref_type") == "pmid":
                                citations.append({
                                    "source": "pubmed",
                                    "id": xref.get("xref_value", {}).get("int", "")
                                })
                        result["citations"] = citations
            
            # Add year for citation formatting
            result["year"] = datetime.now().year
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching {entity_type} {entity_id} from PubChem: {str(e)}")
            return None
    
    async def get_related_entities(self, entity_id: str, entity_type: str = "compound", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get entities related to the specified PubChem entity.
        
        Args:
            entity_id (str): The ID of the entity (CID for compounds, SID for substances, etc.)
            entity_type (str, optional): Type of entity (compound, substance, assay). Defaults to "compound".
            limit (int, optional): Maximum number of results. Defaults to 10.
            
        Returns:
            List[Dict[str, Any]]: List of related entities
        """
        if not self._is_initialized:
            await self.initialize()
            
        # Validate entity type
        if entity_type not in self.entity_types:
            entity_type = "compound"
            
        try:
            results = []
            
            if entity_type == "compound":
                # Strategy 1: Get similar compounds by structure
                url = f"{self.base_url}/compound/cid/{entity_id}/similar/cids/JSON"
                params = {"Threshold": "95", "MaxRecords": str(limit)}
                
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                similar_ids = data.get("IdentifierList", {}).get("CID", [])
                
                # Get details for each similar compound
                for similar_id in similar_ids:
                    entity = await self.get_entity(str(similar_id), "compound")
                    if entity:
                        entity["related_to"] = entity_id
                        entity["relation_type"] = "similar_structure"
                        results.append(entity)
                
                # Strategy 2: Get substances containing this compound
                if len(results) < limit:
                    remaining = limit - len(results)
                    url = f"{self.base_url}/compound/cid/{entity_id}/sids/JSON"
                    
                    try:
                        response = await self.client.get(url)
                        response.raise_for_status()
                        data = response.json()
                        
                        substance_ids = data.get("IdentifierList", {}).get("SID", [])[:remaining]
                        
                        # Get details for each substance
                        for sid in substance_ids:
                            entity = await self.get_entity(str(sid), "substance")
                            if entity:
                                entity["related_to"] = entity_id
                                entity["relation_type"] = "substance_of_compound"
                                results.append(entity)
                    except Exception as e:
                        logger.warning(f"Failed to get substances for compound {entity_id}: {str(e)}")
                
                # Strategy 3: Get bioassays for this compound
                if len(results) < limit:
                    remaining = limit - len(results)
                    url = f"{self.base_url}/compound/cid/{entity_id}/aids/JSON"
                    
                    try:
                        response = await self.client.get(url)
                        response.raise_for_status()
                        data = response.json()
                        
                        assay_ids = data.get("IdentifierList", {}).get("AID", [])[:remaining]
                        
                        # Get details for each assay
                        for aid in assay_ids:
                            entity = await self.get_entity(str(aid), "assay")
                            if entity:
                                entity["related_to"] = entity_id
                                entity["relation_type"] = "assay_testing_compound"
                                results.append(entity)
                    except Exception as e:
                        logger.warning(f"Failed to get assays for compound {entity_id}: {str(e)}")
                
            elif entity_type == "substance":
                # Get compounds contained in this substance
                entity = await self.get_entity(entity_id, "substance")
                if entity and "compound_ids" in entity:
                    compound_ids = entity["compound_ids"][:limit]
                    
                    # Get details for each compound
                    for cid in compound_ids:
                        compound = await self.get_entity(str(cid), "compound")
                        if compound:
                            compound["related_to"] = entity_id
                            compound["relation_type"] = "compound_in_substance"
                            results.append(compound)
            
            elif entity_type == "assay":
                # Get compounds tested in this assay
                url = f"{self.base_url}/assay/aid/{entity_id}/cids/JSON"
                
                try:
                    response = await self.client.get(url)
                    response.raise_for_status()
                    data = response.json()
                    
                    compound_ids = data.get("IdentifierList", {}).get("CID", [])[:limit]
                    
                    # Get details for each compound
                    for cid in compound_ids:
                        compound = await self.get_entity(str(cid), "compound")
                        if compound:
                            compound["related_to"] = entity_id
                            compound["relation_type"] = "compound_tested_in_assay"
                            results.append(compound)
                except Exception as e:
                    logger.warning(f"Failed to get compounds for assay {entity_id}: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting related entities for {entity_type} {entity_id}: {str(e)}")
            return []
    
    def format_citation(self, entity: Dict[str, Any], style: str = "apa") -> str:
        """
        Format citation for a PubChem entity according to the specified citation style.
        
        Args:
            entity (Dict[str, Any]): The entity to cite
            style (str, optional): Citation style (e.g., 'apa', 'mla', 'chicago', 'vancouver'). 
                Defaults to "apa".
                
        Returns:
            str: Formatted citation string
        """
        # Default to APA if style not supported
        if style.lower() not in self.citation_templates:
            style = "apa"
            
        # Extract data needed for citation
        title = entity.get("title", "")
        year = entity.get("year", datetime.now().year)
        entity_type = entity.get("entity_type", "compound")
        entity_id = entity.get("id", "")
        
        # Create identifier string
        identifier = f"PubChem {entity_type.capitalize()} CID {entity_id}"
        
        # Source information
        source = "National Center for Biotechnology Information"
        
        # URL
        url = entity.get("url", f"https://pubchem.ncbi.nlm.nih.gov/{entity_type}/{entity_id}")
        
        # Access date for MLA style
        access_date = datetime.now().strftime("%d %b. %Y")
        
        # Create dictionary with formatted components
        citation_data = {
            "title": title,
            "year": year,
            "identifier": identifier,
            "source": source,
            "url": url,
            "access_date": access_date
        }
        
        # Use the template to format the citation
        citation = self.citation_templates[style].format(**citation_data)
        
        return citation