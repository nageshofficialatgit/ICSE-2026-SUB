// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

contract ElonsMedia {
      string public constant name = "Elons Media";
      string public constant symbol = "EM";
      uint8 public constant decimals = 10;

      mapping(address => uint256) balances;
      mapping(address => mapping(address => uint256)) allowed;

      uint256 totalSupply_;

      event Transfer(address indexed from, address indexed to, uint256 tokens);
      event Approval(address indexed tokenOwner, address indexed spender, uint256 tokens);
      event Burn(address indexed burner, uint256 tokens);

      constructor(uint256 total) {
              totalSupply_ = total;
              balances[msg.sender] = totalSupply_;
      }

      function totalSupply() external view returns (uint256) {
              return totalSupply_;
      }

      function balanceOf(address _tokenOwner) external view returns (uint256) {
              return balances[_tokenOwner];
      }

      function _burn(address _tokenOwner, uint256 _numTokens) internal returns (bool) {
              require(_tokenOwner != address(0), "ERC20: burn from the zero address");
              require(_numTokens <= balances[_tokenOwner], "Insufficient balance");

              totalSupply_ -= _numTokens;
              balances[_tokenOwner] -= _numTokens;
              emit Transfer(_tokenOwner, address(0), _numTokens);
              emit Burn(_tokenOwner, _numTokens);

              return true;
      }

      function burn(uint256 _numTokens) external returns (bool) {
              _burn(msg.sender, _numTokens);
              return true;
      }

      function transfer(address _receiver, uint256 _numTokens) external returns (bool) {
              require(_numTokens <= balances[msg.sender], "Insufficient balance");

              uint256 fee = (_numTokens * 10) / 100;
              uint256 senderAmount = _numTokens - fee;

              balances[msg.sender] -= _numTokens;
              balances[_receiver] += senderAmount;
              emit Transfer(msg.sender, _receiver, senderAmount);
              _burn(msg.sender, fee);

              return true;
      }

      function approve(address delegate, uint256 numTokens) public returns (bool) {
              allowed[msg.sender][delegate] = numTokens;
              emit Approval(msg.sender, delegate, numTokens);
              return true;
      }

      function allowance(address owner, address delegate) public view returns (uint256) {
              return allowed[owner][delegate];
      }

      function transferFrom(address owner, address buyer, uint numTokens) public returns (bool) {
              require(numTokens <= balances[owner], "Insufficient balance");
              require(numTokens <= allowed[owner][msg.sender], "Insufficient allowance");

              balances[owner] -= numTokens;
              allowed[owner][msg.sender] -= numTokens;
              balances[buyer] += numTokens;
              emit Transfer(owner, buyer, numTokens);

              return true;
      }
}