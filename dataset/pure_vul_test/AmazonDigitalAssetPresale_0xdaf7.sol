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
    uint256 public presaleTokenAmount = 100000000000 * 10**2; // 100 млрд токенов с decimals = 2
    bool public presaleActive = true;
    uint256 public totalSold = 0;

    struct Stage {
        uint256 id;
        uint256 bonus; // In percentage, e.g., 20 for 20%
        uint256 price; // Price per token in USD with decimals = 2 (e.g., 4 USD * 10^2 = 400)
        uint256 start;
        uint256 end;
    }
    mapping(uint256 => Stage) public stages;
    uint256 public maxStage = 1;
    uint256 public currentStageId = 1;

    constructor(address _token, address _priceFeed) Ownable(msg.sender) {
        token = IERC20(_token);
        tokenMetadata = IERC20Metadata(_token);
        priceFeed = AggregatorV3Interface(_priceFeed);
        // Начало стадии: 24 марта 2024 года (1711238400)
        // Конец стадии: 24 марта 2025 года (1711238400 + 365 дней * 24 часа * 60 минут * 60 секунд = 1742774400)
        stages[1] = Stage(1, 20, 400, 1711238400, 1742774400); // 20% bonus, 4 USD per token (400 с decimals = 2)
        currentStageId = 1;
    }

    function getEthToUsdPrice() public view returns (int256) {
        (, int256 price, , , ) = priceFeed.latestRoundData();
        return price;
    }

    function convertEthToUsd(uint256 ethAmount) public view returns (uint256) {
        int256 ethToUsdPrice = getEthToUsdPrice();
        uint256 usdAmount = (ethAmount * uint256(ethToUsdPrice)) / (10 ** priceFeed.decimals());
        return usdAmount * 10**2; // Приводим к decimals = 2
    }

    function buyToken(uint256 _amount) public payable {
        require(presaleActive, "Presale is not active!");
        require(_amount > 0, "Please enter minimum token!");
        uint256 _id = getCurrentStageIdActive();
        require(_id > 0, "Stage info not available!");
        uint256 _bonus = stages[_id].bonus;
        uint256 _price = stages[_id].price; // Price in USD with decimals = 2 (e.g., 400 for 4 USD)

        // Calculate cost in USD (with decimals = 2)
        // _amount is in token units (e.g., 229 for 2.29 tokens), _price is in USD (e.g., 400 for 4 USD)
        uint256 _totalPayUsd = (_amount * _price) / 100; // USD with decimals = 2

        // Convert USD to ETH
        uint256 _ethToUsd = convertEthToUsd(1e18); // ETH/USD rate with decimals = 2
        uint256 _totalPayAmount = (_totalPayUsd * 1e18) / _ethToUsd; // Cost in wei

        // Check payment
        require(msg.value >= _totalPayAmount, "Not enough payment!");

        // Calculate tokens and bonus
        uint256 _bonusAmount = (_amount * _bonus) / 100; // Bonus in token units
        uint256 _totalAmount = _amount + _bonusAmount; // Total tokens in token units

        // Check token availability
        require(
            _totalAmount <= token.balanceOf(address(this)),
            "Presale contract doesn't have enough token!"
        );
        require(
            (totalSold + _totalAmount) <= presaleTokenAmount,
            "Presale token amount exceeds!"
        );

        // Transfer tokens to buyer
        require(
            token.transfer(msg.sender, _totalAmount),
            "Failed to transfer token!"
        );

        // Update total sold
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

    function setMaxStage(uint256 _maxStage) public onlyOwner {
        maxStage = _maxStage;
    }

    function setTotalSold(uint256 _totalSold) public onlyOwner {
        totalSold = _totalSold;
    }

    function addStage(
        uint256 _bonus,
        uint256 _price,
        uint256 _start,
        uint256 _end
    ) public onlyOwner {
        uint256 _id = currentStageId + 1;
        require(_id <= maxStage, "Maximum stage exceeds!");
        require(_bonus <= 100, "Bonus should be between 0 and 100");
        require(_start > 0 && _end > 0, "Invalid date!");
        require(_end >= _start + 365 days, "End date must be at least one year after start!");
        currentStageId += 1;
        stages[_id] = Stage(_id, _bonus, _price, _start, _end);
    }

    function setStage(
        uint256 _id,
        uint256 _bonus,
        uint256 _price,
        uint256 _start,
        uint256 _end
    ) public onlyOwner {
        require(stages[_id].id == _id, "ID doesn't exist!");
        require(_bonus <= 100, "Bonus should be between 0 and 100");
        require(_start > 0 && _end > 0, "Invalid date!");
        require(_end >= _start + 365 days, "End date must be at least one year after start!");
        stages[_id] = Stage(_id, _bonus, _price, _start, _end);
    }

    function getCurrentStageIdActive() public view returns (uint256) {
        if (!presaleActive || currentStageId == 0) {
            return 0;
        }
        return 1; // Всегда возвращаем первую стадию
    }

    function withdrawFunds() public onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No funds to withdraw!");
        require(
            payable(msg.sender).send(balance),
            "Failed to withdraw funds!"
        );
    }

    function withdrawTokens(address _to, uint256 _amount) public onlyOwner {
        uint256 _tokenBalance = token.balanceOf(address(this));
        require(_tokenBalance >= _amount, "Exceeds token balance!");
        bool success = token.transfer(_to, _amount);
        require(success, "Failed to transfer token!");
    }
}