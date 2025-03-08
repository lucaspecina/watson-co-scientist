"""
UniProt knowledge provider for protein and gene information.
"""

import os
import logging
import httpx
import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from ..base_provider import DomainKnowledgeProvider

logger = logging.getLogger("co_scientist")

class UniProtProvider(DomainKnowledgeProvider):
    """
    UniProt knowledge provider for protein and gene information.
    
    This provider interfaces with the UniProt API to search and retrieve
    data about proteins, genes, and their functions across all species.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the UniProt provider.
        
        Args:
            config (Dict[str, Any], optional): Configuration for the provider. Defaults to None.
        """
        super().__init__(domain="biology", config=config)
        
        # UniProt API base URL
        self.base_url = "https://rest.uniprot.org"
        
        # HTTP client for API requests
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Citation format templates
        self.citation_templates = {
            "apa": "UniProt Consortium ({year}). {title} ({accession}). {database}. Retrieved from {url}",
            "mla": "UniProt Consortium. \"{title}.\" {database}, {year}, {accession}. {url}. Accessed {access_date}.",
            "chicago": "UniProt Consortium. \"{title}.\" {database}, {year}. {url}.",
            "vancouver": "UniProt Consortium. {title}. {database}. {year}. {accession}. Available from: {url}"
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the UniProt provider.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Test connection to UniProt API with a simple query
            url = f"{self.base_url}/uniprotkb/search"
            params = {
                "query": "insulin",
                "format": "json",
                "size": "1"
            }
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            # Parse response to confirm it's valid
            data = response.json()
            if "results" in data:
                self._is_initialized = True
                logger.info("UniProt provider initialized successfully")
                return True
            else:
                logger.error("UniProt API response did not contain expected structure")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize UniProt provider: {str(e)}")
            return False
    
    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search UniProt for proteins and genes.
        
        Args:
            query (str): The search query
            limit (int, optional): Maximum number of results. Defaults to 10.
            
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        if not self._is_initialized:
            await self.initialize()
            
        try:
            # Prepare search query
            url = f"{self.base_url}/uniprotkb/search"
            params = {
                "query": query,
                "format": "json",
                "size": str(limit),
                "fields": "accession,id,protein_name,gene_names,organism_name,cc_function,cc_catalytic_activity,sequence,xref_pdb,go,annotation_score"
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Process results
            results = []
            
            for item in data.get("results", []):
                # Extract basic information
                accession = item.get("accession", "")
                entry_name = item.get("id", "")
                
                # Extract protein name
                protein_name = ""
                if "protein" in item and "recommendedName" in item["protein"]:
                    protein_name = item["protein"]["recommendedName"].get("fullName", {}).get("value", "")
                elif "protein" in item and "submittedName" in item["protein"]:
                    if isinstance(item["protein"]["submittedName"], list):
                        protein_name = item["protein"]["submittedName"][0].get("fullName", {}).get("value", "")
                    else:
                        protein_name = item["protein"]["submittedName"].get("fullName", {}).get("value", "")
                
                # Extract gene names
                gene_names = []
                if "genes" in item:
                    for gene in item["genes"]:
                        if "geneName" in gene:
                            gene_names.append(gene["geneName"].get("value", ""))
                
                # Extract organism
                organism = ""
                if "organism" in item:
                    organism = item["organism"].get("scientificName", "")
                
                # Extract function
                function = ""
                if "comments" in item:
                    for comment in item["comments"]:
                        if comment.get("commentType") == "FUNCTION":
                            function = comment.get("text", [{}])[0].get("value", "")
                
                # Extract GO terms
                go_terms = []
                if "dbReferences" in item:
                    for ref in item["dbReferences"]:
                        if ref.get("type") == "GO":
                            term = {
                                "id": ref.get("id", ""),
                                "term": ref.get("properties", {}).get("term", "")
                            }
                            go_terms.append(term)
                
                # Extract PDB IDs
                pdb_ids = []
                if "dbReferences" in item:
                    for ref in item["dbReferences"]:
                        if ref.get("type") == "PDB":
                            pdb_ids.append(ref.get("id", ""))
                
                # Create result entry
                result = {
                    "id": accession,
                    "accession": accession,
                    "entry_name": entry_name,
                    "title": protein_name or entry_name,
                    "protein_name": protein_name,
                    "gene_names": gene_names,
                    "organism": organism,
                    "function": function,
                    "go_terms": go_terms[:10],  # Limit to top 10 GO terms
                    "pdb_ids": pdb_ids[:5],  # Limit to top 5 PDB IDs
                    "url": f"https://uniprot.org/uniprotkb/{accession}",
                    "source": "uniprot",
                    "provider": "uniprot",
                    "database": "UniProtKB"
                }
                
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"Error searching UniProt: {str(e)}")
            return []
    
    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed information about a specific UniProt entry.
        
        Args:
            entity_id (str): UniProt accession number (e.g., P01308 for human insulin)
            
        Returns:
            Optional[Dict[str, Any]]: Entry information or None if not found
        """
        if not self._is_initialized:
            await self.initialize()
            
        try:
            # Get entry by accession
            url = f"{self.base_url}/uniprotkb/{entity_id}"
            params = {
                "format": "json",
                "fields": "accession,id,protein_name,gene_names,organism_name,cc_function,cc_subcellular_location,cc_disease,cc_catalytic_activity,sequence,xref_pdb,go,annotation_score,ft_active_site,ft_binding,ft_domain,cc_interaction"
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            item = response.json()
            
            # Extract basic information
            accession = item.get("accession", "")
            entry_name = item.get("id", "")
            
            # Extract protein name
            protein_name = ""
            if "protein" in item and "recommendedName" in item["protein"]:
                protein_name = item["protein"]["recommendedName"].get("fullName", {}).get("value", "")
            elif "protein" in item and "submittedName" in item["protein"]:
                if isinstance(item["protein"]["submittedName"], list):
                    protein_name = item["protein"]["submittedName"][0].get("fullName", {}).get("value", "")
                else:
                    protein_name = item["protein"]["submittedName"].get("fullName", {}).get("value", "")
            
            # Extract gene names
            gene_names = []
            if "genes" in item:
                for gene in item["genes"]:
                    if "geneName" in gene:
                        gene_names.append(gene["geneName"].get("value", ""))
            
            # Extract organism
            organism = ""
            if "organism" in item:
                organism = item["organism"].get("scientificName", "")
            
            # Extract function
            function = ""
            if "comments" in item:
                for comment in item["comments"]:
                    if comment.get("commentType") == "FUNCTION":
                        function = comment.get("text", [{}])[0].get("value", "")
            
            # Extract subcellular location
            subcellular_location = ""
            if "comments" in item:
                for comment in item["comments"]:
                    if comment.get("commentType") == "SUBCELLULAR LOCATION":
                        subcellular_location = comment.get("subcellularLocations", [{}])[0].get("location", {}).get("value", "")
            
            # Extract disease associations
            diseases = []
            if "comments" in item:
                for comment in item["comments"]:
                    if comment.get("commentType") == "DISEASE":
                        disease = {
                            "name": comment.get("disease", {}).get("diseaseName", {}).get("value", ""),
                            "description": comment.get("text", [{}])[0].get("value", "")
                        }
                        diseases.append(disease)
            
            # Extract catalytic activity
            catalytic_activity = []
            if "comments" in item:
                for comment in item["comments"]:
                    if comment.get("commentType") == "CATALYTIC ACTIVITY":
                        for reaction in comment.get("reaction", []):
                            catalytic_activity.append(reaction.get("name", ""))
            
            # Extract sequence
            sequence = ""
            sequence_length = 0
            if "sequence" in item:
                sequence = item["sequence"].get("value", "")
                sequence_length = item["sequence"].get("length", 0)
            
            # Extract GO terms
            go_terms = []
            if "dbReferences" in item:
                for ref in item["dbReferences"]:
                    if ref.get("type") == "GO":
                        term = {
                            "id": ref.get("id", ""),
                            "term": ref.get("properties", {}).get("term", ""),
                            "category": ref.get("properties", {}).get("category", "")
                        }
                        go_terms.append(term)
            
            # Extract PDB IDs
            pdb_ids = []
            if "dbReferences" in item:
                for ref in item["dbReferences"]:
                    if ref.get("type") == "PDB":
                        pdb_ids.append(ref.get("id", ""))
            
            # Extract protein features
            features = []
            if "features" in item:
                for feature in item["features"]:
                    if feature.get("type") in ["ACTIVE_SITE", "BINDING", "DOMAIN"]:
                        feat = {
                            "type": feature.get("type", ""),
                            "description": feature.get("description", ""),
                            "start": feature.get("location", {}).get("start", {}).get("value", ""),
                            "end": feature.get("location", {}).get("end", {}).get("value", "")
                        }
                        features.append(feat)
            
            # Extract interactions
            interactions = []
            if "comments" in item:
                for comment in item["comments"]:
                    if comment.get("commentType") == "INTERACTION":
                        for interaction in comment.get("interactions", []):
                            interactor = {
                                "id": interaction.get("interactant", {}).get("uniProtkbAccession", ""),
                                "gene": interaction.get("interactant", {}).get("geneName", ""),
                                "experiments": interaction.get("numberOfExperiments", 0)
                            }
                            interactions.append(interactor)
            
            # Create result
            result = {
                "id": accession,
                "accession": accession,
                "entry_name": entry_name,
                "title": protein_name or entry_name,
                "protein_name": protein_name,
                "gene_names": gene_names,
                "organism": organism,
                "function": function,
                "subcellular_location": subcellular_location,
                "diseases": diseases,
                "catalytic_activity": catalytic_activity,
                "sequence": sequence,
                "sequence_length": sequence_length,
                "go_terms": go_terms,
                "pdb_ids": pdb_ids,
                "features": features,
                "interactions": interactions[:20],  # Limit to top 20 interactions
                "url": f"https://uniprot.org/uniprotkb/{accession}",
                "source": "uniprot",
                "provider": "uniprot",
                "database": "UniProtKB",
                "year": datetime.now().year  # For citation
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching UniProt entry {entity_id}: {str(e)}")
            return None
    
    async def get_related_entities(self, entity_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get proteins related to the specified UniProt entry.
        
        Args:
            entity_id (str): UniProt accession number
            limit (int, optional): Maximum number of results. Defaults to 10.
            
        Returns:
            List[Dict[str, Any]]: List of related proteins
        """
        if not self._is_initialized:
            await self.initialize()
            
        try:
            # First get the protein details
            protein = await self.get_entity(entity_id)
            if not protein:
                return []
            
            results = []
            strategies = []
            
            # Strategy 1: Get proteins that interact with this protein
            if "interactions" in protein and protein["interactions"]:
                strategies.append("interactions")
                
                for interaction in protein["interactions"][:limit]:
                    if "id" in interaction and interaction["id"]:
                        related = await self.get_entity(interaction["id"])
                        if related:
                            related["related_to"] = entity_id
                            related["relation_type"] = "interaction"
                            results.append(related)
                            
                            if len(results) >= limit:
                                return results
            
            # Strategy 2: Get proteins with similar function (based on GO terms)
            if len(results) < limit and "go_terms" in protein and protein["go_terms"]:
                strategies.append("go_terms")
                
                # Use molecular function GO terms for the search
                function_go_terms = [term["id"] for term in protein["go_terms"] 
                                  if term.get("category") == "MOLECULAR_FUNCTION"]
                
                if function_go_terms:
                    # Take up to 3 GO terms to avoid too narrow searches
                    go_query = " OR ".join([f"go:{term}" for term in function_go_terms[:3]])
                    
                    # Add organism constraint if available
                    if protein.get("organism"):
                        organism = protein["organism"].lower().split()[0]  # Get genus
                        go_query = f"({go_query}) AND organism:{organism}"
                    
                    # Search for proteins with similar function
                    go_results = await self.search(go_query, limit=limit*2)
                    
                    # Filter out the original protein and those already in results
                    existing_ids = {entity_id} | {r["id"] for r in results}
                    go_results = [r for r in go_results if r["id"] not in existing_ids]
                    
                    # Add relation metadata
                    for result in go_results:
                        result["related_to"] = entity_id
                        result["relation_type"] = "similar_function"
                    
                    # Add results
                    results.extend(go_results[:limit - len(results)])
                    
                    if len(results) >= limit:
                        return results
            
            # Strategy 3: Get proteins from the same gene family
            if len(results) < limit and protein.get("gene_names"):
                strategies.append("gene_family")
                
                # Get the first gene name
                gene_name = protein["gene_names"][0]
                
                # Search for proteins in the same gene family
                gene_query = f"gene:{gene_name}* AND reviewed:true"
                
                # Add organism constraint if available
                if protein.get("organism"):
                    organism = protein["organism"].lower().split()[0]  # Get genus
                    gene_query = f"{gene_query} AND organism:{organism}"
                
                # Search for proteins in the same gene family
                gene_results = await self.search(gene_query, limit=limit*2)
                
                # Filter out the original protein and those already in results
                existing_ids = {entity_id} | {r["id"] for r in results}
                gene_results = [r for r in gene_results if r["id"] not in existing_ids]
                
                # Add relation metadata
                for result in gene_results:
                    result["related_to"] = entity_id
                    result["relation_type"] = "gene_family"
                
                # Add results
                results.extend(gene_results[:limit - len(results)])
            
            # Add the strategies used to the results metadata
            for result in results:
                result["relation_strategies"] = strategies
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error finding related proteins for {entity_id}: {str(e)}")
            return []
    
    def format_citation(self, entity: Dict[str, Any], style: str = "apa") -> str:
        """
        Format citation for a UniProt entry according to the specified citation style.
        
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
        if entity.get("protein_name"):
            title = f"{entity['protein_name']} ({', '.join(entity.get('gene_names', [])[:3])})"
        
        accession = entity.get("accession", "")
        database = entity.get("database", "UniProtKB")
        year = entity.get("year", datetime.now().year)
        url = entity.get("url", f"https://uniprot.org/uniprotkb/{accession}")
        
        # Access date for MLA style
        access_date = datetime.now().strftime("%d %b. %Y")
        
        # Create dictionary with formatted components
        citation_data = {
            "title": title,
            "accession": accession,
            "database": database,
            "year": year,
            "url": url,
            "access_date": access_date
        }
        
        # Use the template to format the citation
        citation = self.citation_templates[style].format(**citation_data)
        
        return citation