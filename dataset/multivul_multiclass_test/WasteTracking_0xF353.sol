// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract WasteTracking {
    struct PickupRequest {
        string pr_id;  // Pickup Request ID
        string fo_id;  // Farm Owner ID
        uint256 ftcw;  // Farm to Collector Weight
        uint256 ctfw;  // Collector to Facility Weight (Updated later)
        string eka;    // EKA Code
        string wt_id;  // Waste Transporter ID
        uint256 f_id;  // Facility ID
        string status; // Status of update ("Pending" or "Updated")
    }

    mapping(string => PickupRequest) public pickupRequests;
    address public owner;

    event PickupRecorded(
        string pr_id,
        string fo_id,
        uint256 ftcw,
        uint256 ctfw,
        string eka,
        string wt_id,
        uint256 f_id,
        string status, // Status ("Pending" or "Updated")
        uint256 timestamp
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function recordPickup(
        string memory _pr_id,
        string memory _fo_id,
        uint256 _ftcw,
        uint256 _ctfw,
        string memory _eka,
        string memory _wt_id,
        uint256 _f_id
    ) public onlyOwner {
        require(bytes(pickupRequests[_pr_id].pr_id).length == 0, "Pickup Request already exists!");
        
        pickupRequests[_pr_id] = PickupRequest(
            string(abi.encodePacked(_pr_id)), 
            string(abi.encodePacked(_fo_id)), 
            _ftcw, 
            _ctfw, 
            string(abi.encodePacked(_eka)), 
            string(abi.encodePacked(_wt_id)), 
            _f_id, 
            "Pending"
        );
        
        emit PickupRecorded(
            _pr_id, 
            _fo_id, 
            _ftcw, 
            _ctfw, 
            _eka, 
            _wt_id, 
            _f_id, 
            "Pending",
            block.timestamp
        );
    }

    function updateFacilityWeight(string memory _pr_id, uint256 _new_ctfw) public onlyOwner {
        require(bytes(pickupRequests[_pr_id].pr_id).length != 0, "Pickup Request does not exist!");
        require(keccak256(bytes(pickupRequests[_pr_id].status)) != keccak256(bytes("Updated")), "Facility weight already updated!");
        
        pickupRequests[_pr_id].ctfw = _new_ctfw;
        pickupRequests[_pr_id].status = "Updated";
        
        emit PickupRecorded(
            string(abi.encodePacked(_pr_id)), 
            string(abi.encodePacked(pickupRequests[_pr_id].fo_id)), 
            pickupRequests[_pr_id].ftcw, 
            _new_ctfw, 
            string(abi.encodePacked(pickupRequests[_pr_id].eka)), 
            string(abi.encodePacked(pickupRequests[_pr_id].wt_id)), 
            pickupRequests[_pr_id].f_id, 
            "Updated",
            block.timestamp
        );
    }

    function getPickupRequest(string memory _pr_id) public view onlyOwner returns (
        string memory,
        string memory,
        uint256,
        uint256,
        string memory,
        string memory,
        uint256,
        string memory
    ) {
        require(bytes(pickupRequests[_pr_id].pr_id).length != 0, "Pickup Request not found!");

        PickupRequest memory request = pickupRequests[_pr_id];
        return (
            request.pr_id,
            request.fo_id,
            request.ftcw,
            request.ctfw,
            request.eka,
            request.wt_id,
            request.f_id,
            request.status
        );
    }
}