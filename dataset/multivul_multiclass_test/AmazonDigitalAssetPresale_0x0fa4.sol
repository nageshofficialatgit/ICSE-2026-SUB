// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }

    function _contextSuffixLength() internal view virtual returns (uint256) {
        return 0;
    }
}

abstract contract Ownable is Context {
    address private _owner;

    error OwnableUnauthorizedAccount(address account);

    error OwnableInvalidOwner(address owner);

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor(address initialOwner) {
        if (initialOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(initialOwner);
    }

    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    function _checkOwner() internal view virtual {
        if (owner() != _msgSender()) {
            revert OwnableUnauthorizedAccount(_msgSender());
        }
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        if (newOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(newOwner);
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

interface IERC20 {

    event Transfer(address indexed from, address indexed to, uint256 value);

    event Approval(address indexed owner, address indexed spender, uint256 value);

    function totalSupply() external view returns (uint256);

    function balanceOf(address account) external view returns (uint256);

    function transfer(address to, uint256 value) external returns (bool);

    function allowance(address owner, address spender) external view returns (uint256);

    function approve(address spender, uint256 value) external returns (bool);

    function transferFrom(address from, address to, uint256 value) external returns (bool);
}

interface IERC20Metadata is IERC20 {

    function name() external view returns (string memory);

    function symbol() external view returns (string memory);

    function decimals() external view returns (uint8);
}

interface AggregatorV3Interface {
  function decimals() external view returns (uint8);

  function description() external view returns (string memory);

  function version() external view returns (uint256);

  function getRoundData(uint80 _roundId)
    external
    view
    returns (
      uint80 roundId,
      int256 answer,
      uint256 startedAt,
      uint256 updatedAt,
      uint80 answeredInRound
    );

  function latestRoundData()
    external
    view
    returns (
      uint80 roundId,
      int256 answer,
      uint256 startedAt,
      uint256 updatedAt,
      uint80 answeredInRound
    );
}

contract AmazonDigitalAssetPresale is Ownable {
    IERC20 public token;
    IERC20Metadata public tokenMetadata;
    AggregatorV3Interface public priceFeed;
    uint256 public presaleTokenAmount = 1000000000;
    bool public presaleActive = true;
    uint256 public totalSold = 0;

    struct Stage {
        uint256 id;
        uint256 bonus;
        uint256 price;
        uint256 start;
        uint256 end;
    }
    mapping(uint256 => Stage) public stages;
    uint256 public currentStageId = 1;

    constructor(address _token, address _priceFeed) Ownable(msg.sender) {
        token = IERC20(_token);
        tokenMetadata = IERC20Metadata(_token);
        priceFeed = AggregatorV3Interface(_priceFeed);
        stages[1] = Stage(1, 20, 4000000000000000000, block.timestamp - 7 days, block.timestamp + 365 days);
    }

    function getEthToUsdPrice() public view returns (int256) {
        (, int256 price, , , ) = priceFeed.latestRoundData();
        return price;
    }

    function convertEthToUsd(uint256 ethAmount) public view returns (uint256) {
        int256 ethToUsdPrice = getEthToUsdPrice();
        return (ethAmount * uint256(ethToUsdPrice)) / (10**priceFeed.decimals());
    }

    function buyToken(uint256 _amount) public payable {
        require(presaleActive, "Presale is not active!");
        require(_amount > 0, "Please enter minimum token!");
        uint256 _id = getCurrentStageIdActive();
        require(_id > 0, "Stage info not available!");
        uint256 _bonus = stages[_id].bonus;
        uint256 _price = stages[_id].price;
        uint256 _start = stages[_id].start;
        uint256 _end = stages[_id].end;
        require(_start <= block.timestamp, "Presale has not started yet!");
        require(_end >= block.timestamp, "Presale end!");
        uint256 _totalPayUsd = _amount * _price;
        uint256 _ethToUsd = convertEthToUsd(1e18);
        uint256 _totalPayAmount = _totalPayUsd / _ethToUsd;
        require(msg.value >= _totalPayAmount, "Not enough payment!");
        uint256 _weiAmount = _amount * 1e4;
        uint256 _bonusAmount = (_weiAmount * _bonus) / 100;
        uint256 _totalAmount = _weiAmount + _bonusAmount;
        require(_totalAmount <= token.balanceOf(address(this)), "Not enough tokens in contract!");
        require((totalSold + _totalAmount) <= presaleTokenAmount, "Presale token amount exceeds!");
        require(token.transfer(msg.sender, _totalAmount), "Failed to transfer token!");
        totalSold += _totalAmount;
    }

    function setToken(address _token) public onlyOwner {
        require(_token != address(0), "Token is zero address!");
        token = IERC20(_token);
        tokenMetadata = IERC20Metadata(_token);
    }

    function setPriceFeed(address _priceFeed) public onlyOwner {
        require(_priceFeed != address(0), "Price feed is zero address!");
        priceFeed = AggregatorV3Interface(_priceFeed);
    }

    function setPresaleTokenAmount(uint256 _amount) public onlyOwner {
        presaleTokenAmount = _amount;
    }

    function flipPresaleActive() public onlyOwner {
        presaleActive = !presaleActive;
    }

    function setTotalSold(uint256 _totalSold) public onlyOwner {
        totalSold = _totalSold;
    }

    function getCurrentStageIdActive() public view returns (uint256) {
        if (currentStageId == 0) return 0;
        if (block.timestamp >= stages[1].start && block.timestamp <= stages[1].end) return 1;
        return 0;
    }

    function withdrawFunds() public onlyOwner {
        require(payable(msg.sender).send(address(this).balance), "Failed withdraw!");
    }

    function withdrawTokens(address _to, uint256 _amount) public onlyOwner {
        uint256 _tokenBalance = token.balanceOf(address(this));
        require(_tokenBalance >= _amount, "Exceeds token balance!");
        require(token.transfer(_to, _amount), "Failed to transfer token!");
    }
}