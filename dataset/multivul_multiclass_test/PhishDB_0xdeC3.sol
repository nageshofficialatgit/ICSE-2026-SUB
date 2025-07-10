// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract PhishDB {
    string[] public reportedURLs;
    mapping(string => bool) public isReported;

    function reportURL(string memory _url) public {
        require(!isReported[_url], "URL already reported");
        reportedURLs.push(_url);
        isReported[_url] = true;
    }
}