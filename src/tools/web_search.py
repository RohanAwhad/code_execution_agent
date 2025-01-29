import dataclasses
import os
import requests
from typing import List

@dataclasses.dataclass
class SearchResult:
  """
  Dataclass to represent the search results from Brave Search API.

  :param title: The title of the search result.
  :param url: The URL of the search result.
  :param description: A brief description of the search result.
  :param extra_snippets: Additional snippets related to the search result.
  """
  title: str
  url: str
  description: str
  extra_snippets: list

  def __str__(self) -> str:
    """
    Returns a string representation of the search result.

    :return: A string representation of the search result.
    """
    return (
        f"Title: {self.title}\n"
        f"URL: {self.url}\n"
        f"Description: {self.description}\n"
        f"Extra Snippets: {', '.join(self.extra_snippets)}"
    )


def search_brave(query: str, count: int = 10) -> List[SearchResult]:
  """
  Searches the web using Brave Search API and returns structured search results.

  :param query: The search query string.
  :param count: The number of search results to return.
  :return: A list of SearchResult objects containing the search results.
  """
  if not query:
    return []
  url = "https://api.search.brave.com/res/v1/web/search"
  headers = {
      "Accept": "application/json",
      "X-Subscription-Token": os.environ['BRAVE_SEARCH_AI_API_KEY']
  }
  params = {
      "q": query,
      "count": count
  }

  response = requests.get(url, headers=headers, params=params)
  response.raise_for_status()  # Raises an exception for HTTP errors
  results_json = response.json()

  results = []
  for item in results_json.get('web', {}).get('results', []):
    result = SearchResult(
        title=item.get('title', ''),
        url=item.get('url', ''),
        description=item.get('description', ''),
        extra_snippets=item.get('extra_snippets', [])
    )
    results.append(result)

  print('Search results')
  print(results)
  return results
