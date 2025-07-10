// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract FavoriteToken {
    string public favorite_token = "btc";
    string public favorite_token2 = "eth";
    string public favorite_token3 = "doge";
    string public favorite_token4 = "sol";
    string public favorite_token5 = "xrp";

    function setFavoriteToken(string memory _favorite_token) public {
        favorite_token = _favorite_token;
    }

    function setFavoriteToken2(string memory _favorite_token) public {
        favorite_token2 = _favorite_token;
    }

    function setFavoriteToken3(string memory _favorite_token) public {
        favorite_token3 = _favorite_token;
    }

    function setFavoriteToken4(string memory _favorite_token) public {
        favorite_token4 = _favorite_token;
    }

    function setFavoriteToken5(string memory _favorite_token) public {
        favorite_token5 = _favorite_token;
    }
}